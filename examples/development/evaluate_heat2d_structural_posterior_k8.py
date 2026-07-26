#!/usr/bin/env python3
"""Preregister and execute a sealed Heat2D K8 allocator kill test.

``prepare`` authenticates the completed source campaign, reconstructs all six
finite K8 slates, applies the frozen structural-posterior policy, writes the
exact 24-candidate evaluation plan, and seals it without a provider or PDE
call.  ``execute`` accepts only that sealed plan, evaluates the previously
unselected candidates with the unchanged direct-v3 boundary, and adjudicates
the preregistered promotion/kill rule.

The endpoint is a fixed-parent, fixed-K8 local allocator diagnostic.  It is not
a campaign counterfactual or an AgentEvolve efficacy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.typed_json import (  # noqa: E402
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.objective_resolution.fixed_grid import (  # noqa: E402
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineHypervolume2DSpec,
    AffineHypervolumeSnapshot2D,
    AffineObjectiveAxis,
)
from agent_evolve.policies.selection.calibrated_slate_codec import (  # noqa: E402
    decode_slate_allocation_request_record,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    StructuralPosteriorSlatePolicy,
)
from agent_evolve.ports.objective_resolution import (  # noqa: E402
    ObjectiveResolutionRequest,
    resolve_objectives,
)
from agent_evolve.ports.variation_catalog import (  # noqa: E402
    bind_finite_variation_catalog,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (  # noqa: E402
    Heat2DFiniteVariationCatalog,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (  # noqa: E402
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    Heat2DMultiObjectiveV1Problem,
)
from examples.benchmarks.heat2d_constructive.phenotype_identity import (  # noqa: E402
    Heat2DPhenotypeIdentityPolicy,
)
from examples.benchmarks.heat2d_constructive.problem_def import (  # noqa: E402
    Heat2DDirectV3Settings,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


DEFAULT_SOURCE_RUN = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/generic_campaign"
    / "heat_calibrated_anchored_g6_live_deepseek_r3_20260716"
)
DEFAULT_PREREGISTRATION = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts"
    / "256_structural_posterior_allocator_transfer_and_heat_k8_preregistration.md"
)
OBJECTIVE_IDS = tuple(sorted((THERMAL_OBJECTIVE_NAME, MATERIAL_OBJECTIVE_NAME)))
EXPECTED_SOURCE_FINALIZATION = (
    "34713c7acaa2f4f25281fafec9e495b73628eef904d3de2423634b0c6dae84ca"
)
EXPECTED_SOURCE_CONTENT = (
    "efcef76a16282e8e3447651f38ca7f35ed85631f3043bcf327f465ad94f55f1d"
)
PROMOTION_UNIFORM_MULTIPLE = 1.10
PROMOTION_HISTORICAL_MULTIPLE = 1.05
PROMOTION_WAVE_WINS = 4
LEGACY_KILL_TEST_MODE = "legacy_preregistered_kill_test"
DESCRIPTIVE_SUPPORT_MODE = "descriptive_support_completion"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"{name} must be an exact JSON object")
    return value


def _array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{name} must be an exact JSON array")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return _object(json.load(stream), name=path.name)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign_events(source_run: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in read_jsonl(source_run / "campaign_events.jsonl"):
        result.append(
            _object(
                row.get("authenticated_campaign_event"),
                name="authenticated campaign event",
            )
        )
    if not result:
        raise RuntimeError("source campaign contains no authenticated events")
    return result


def _candidate_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def add(candidate_value: object) -> None:
        candidate = _object(candidate_value, name="candidate")
        candidate_id = candidate.get("candidate_id")
        if type(candidate_id) is not str or not candidate_id:
            raise RuntimeError("candidate omitted its exact ID")
        prior = result.get(candidate_id)
        if prior is not None and prior != candidate:
            raise RuntimeError("candidate ID resolves to inconsistent records")
        result[candidate_id] = candidate

    starts = [value for value in events if value.get("kind") == "execution_started"]
    if len(starts) != 1:
        raise RuntimeError("source campaign must contain one execution start")
    start_payload = _object(starts[0].get("payload"), name="start payload")
    start_receipt = _object(
        start_payload.get("start_receipt"), name="start receipt"
    )
    for seed_value in _array(start_receipt.get("seed_receipts"), name="seed receipts"):
        seed = _object(seed_value, name="seed receipt")
        evidence = _object(seed.get("evidence"), name="seed evidence")
        add(evidence.get("candidate"))
    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _object(event.get("payload"), name="stage payload")
        stage = _object(payload.get("stage_receipt"), name="stage receipt")
        result_record = _object(stage.get("result"), name="stage result")
        for candidate in _array(result_record.get("candidates"), name="candidates"):
            add(candidate)
    return result


def _objective_values(candidate: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row_value in _array(candidate.get("objectives"), name="candidate objectives"):
        row = _object(row_value, name="candidate objective")
        metric_id = row.get("metric_id")
        value_hex = row.get("value_hex")
        if type(metric_id) is not str or type(value_hex) is not str:
            raise RuntimeError("candidate objective is malformed")
        result[metric_id] = float.fromhex(value_hex)
    if set(result) != set(OBJECTIVE_IDS):
        raise RuntimeError("candidate differs from exact Heat2D objectives")
    if (
        candidate.get("valid") is not True
        or candidate.get("operator_compliant") is not True
        or candidate.get("evidence_compliant") is not True
    ):
        raise RuntimeError("source candidate is not valid and evidence-compliant")
    return result


def _prompt_frame(request_text: object) -> dict[str, Any]:
    if type(request_text) is not str or "{" not in request_text:
        raise RuntimeError("selector request omitted its exact JSON frame")
    suffix = request_text[request_text.index("{") :]
    value, end = json.JSONDecoder().raw_decode(suffix)
    if not suffix[end:].startswith("\n"):
        raise RuntimeError("selector request has a foreign prompt-frame boundary")
    return _object(value, name="selector prompt frame")


def _point(record_value: object) -> dict[str, float]:
    result: dict[str, float] = {}
    for cell_value in _array(record_value, name="objective point"):
        cell = _array(cell_value, name="objective point cell")
        if len(cell) != 2 or any(type(value) is not str for value in cell):
            raise RuntimeError("objective point cell is malformed")
        result[cell[0]] = float.fromhex(cell[1])
    return result


def _affine_snapshot(receipt_value: object) -> AffineHypervolumeSnapshot2D:
    receipt = _object(receipt_value, name="affine snapshot receipt")
    spec_record = _object(receipt.get("spec"), name="affine specification")
    axes: list[AffineObjectiveAxis] = []
    for axis_value in _array(spec_record.get("axes"), name="affine axes"):
        axis = _object(axis_value, name="affine axis")
        axes.append(
            AffineObjectiveAxis(
                metric_id=axis["metric_id"],
                goal=axis["goal"],
                ideal=float.fromhex(axis["ideal_hex"]),
                reference=float.fromhex(axis["reference_hex"]),
            )
        )
    if len(axes) != 2:
        raise RuntimeError("Heat2D affine snapshot must have exactly two axes")
    spec = AffineHypervolume2DSpec(
        axes=(axes[0], axes[1]),
        reference_provenance=spec_record["reference_provenance"],
    )
    if spec.to_record() != spec_record:
        raise RuntimeError("reconstructed affine specification differs from trace")
    snapshot = AffineHypervolumeSnapshot2D.create(
        spec=spec,
        archive_points=tuple(
            _point(value)
            for value in _array(
                receipt.get("raw_archive_points"), name="raw archive points"
            )
        ),
    )
    if snapshot.to_record() != receipt:
        raise RuntimeError("reconstructed affine snapshot differs from trace")
    return snapshot


def _archive_receipts(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "archive_utility_frozen":
            continue
        payload = _object(event.get("payload"), name="archive utility payload")
        utility = _object(payload.get("archive_utility"), name="archive utility")
        generation = utility.get("generation")
        receipt = _object(utility.get("snapshot_receipt"), name="snapshot receipt")
        if type(generation) is not int or generation in result:
            raise RuntimeError("archive generation is not exact and unique")
        _affine_snapshot(receipt)
        result[generation] = receipt
    return result


def _allocation_from_audit(
    audit: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    plaintext = _object(audit.get("plaintext_audit"), name="plaintext audit")
    response_text = plaintext.get("response_text")
    if type(response_text) is not str:
        raise RuntimeError("selector audit omitted trusted response text")
    response = _object(json.loads(response_text), name="selector response")
    supplemental = _object(
        response.get("supplemental_selector_audit"), name="supplemental audit"
    )
    if (
        supplemental.get("decision_sha256") != audit.get("decision_sha256")
        or supplemental.get("request_sha256") != audit.get("request_sha256")
    ):
        raise RuntimeError("supplemental audit does not join its selector audit")
    payload = _object(supplemental.get("payload"), name="supplemental payload")
    allocation = _object(payload.get("allocation"), name="historical allocation")
    request = decode_slate_allocation_request_record(allocation.get("request"))
    return allocation, request


def _build_precommit(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates = _candidate_index(events)
    candidate_by_configuration = {
        value["configuration_sha256"]: value for value in candidates.values()
    }
    archives = _archive_receipts(events)
    policy = StructuralPosteriorSlatePolicy()
    phenotype_identity = Heat2DPhenotypeIdentityPolicy(resolution=1001)
    waves: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    selected_occurrences = 0
    prior_occurrences = 0
    support_configurations: set[str] = set()

    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _object(event.get("payload"), name="stage payload")
        stage = _object(payload.get("stage_receipt"), name="stage receipt")
        generation = stage.get("generation")
        if type(generation) is not int or generation % 2 == 0:
            continue
        result_record = _object(stage.get("result"), name="stage result")
        wave_receipts = {
            value["selection_call_id"]: value
            for value in (
                _object(item, name="portfolio wave receipt")
                for item in _array(
                    result_record.get("portfolio_wave_receipts"),
                    name="portfolio wave receipts",
                )
            )
        }
        audits = _array(stage.get("selector_audits"), name="selector audits")
        if len(audits) != 2 or len(wave_receipts) != 2:
            raise RuntimeError("each odd generation must contain two lanes")
        archive_receipt = archives.get(generation)
        if archive_receipt is None:
            raise RuntimeError("portfolio generation omitted affine snapshot")

        for audit_value in audits:
            audit = _object(audit_value, name="selector audit")
            call_id = audit.get("selector_call_id")
            if type(call_id) is not str or call_id not in wave_receipts:
                raise RuntimeError("selector audit has no exact wave receipt")
            wave_receipt = wave_receipts[call_id]
            plaintext = _object(audit.get("plaintext_audit"), name="plaintext audit")
            frame = _prompt_frame(plaintext.get("request_text"))
            context = _object(frame.get("context"), name="selector context")
            parent_configuration = _object(
                context.get("parent_configuration"), name="parent configuration"
            )
            frozen_parent = freeze_json(parent_configuration)
            parent_sha256 = typed_json_sha256(frozen_parent)
            if (
                parent_sha256 != context.get("parent_configuration_sha256")
                or parent_sha256 != wave_receipt.get("parent_configuration_sha256")
            ):
                raise RuntimeError("prompt and wave receipt differ on parent identity")
            contract = bind_finite_variation_catalog(
                Heat2DFiniteVariationCatalog(), frozen_parent
            )
            prompt_contract = _object(
                frame.get("finite_variation_contract"), name="prompt contract"
            )
            if prompt_contract.get("parent_configuration_sha256") != parent_sha256:
                raise RuntimeError("prompt finite contract names a foreign parent")

            parent_candidate_id = wave_receipt.get("parent_candidate_id")
            if (
                type(parent_candidate_id) is not str
                or parent_candidate_id not in candidates
                or candidates[parent_candidate_id].get("configuration_sha256")
                != parent_sha256
            ):
                raise RuntimeError("wave parent does not join source candidate")
            parent_objectives = _objective_values(candidates[parent_candidate_id])
            allocation, request = _allocation_from_audit(audit)
            decision = policy.select(request)
            structural_selected = {value.option_id for value in decision.selected}
            historical_selected = {
                _object(value, name="historical selection")["option_id"]
                for value in _array(
                    allocation.get("selected"), name="historical selections"
                )
            }
            if len(historical_selected) != 4:
                raise RuntimeError("historical allocator did not select exact K4")
            selected_outcomes: dict[str, tuple[dict[str, float], str]] = {}
            for attribution_value in _array(
                wave_receipt.get("action_attributions"),
                name="action attributions",
            ):
                attribution = _object(attribution_value, name="action attribution")
                member = _object(
                    attribution.get("selected_member"), name="selected member"
                )
                option_id = member.get("option_id")
                candidate_id = attribution.get("candidate_id")
                if type(option_id) is not str or type(candidate_id) is not str:
                    raise RuntimeError("selected attribution is malformed")
                source_candidate = candidates.get(candidate_id)
                if source_candidate is None:
                    raise RuntimeError("selected attribution names unknown candidate")
                selected_outcomes[option_id] = (
                    _objective_values(source_candidate),
                    candidate_id,
                )
            if set(selected_outcomes) != historical_selected:
                raise RuntimeError("selected outcomes differ from historical K4")

            ordered: list[dict[str, Any]] = []
            for member in sorted(request.slate.members, key=lambda value: value.model_rank):
                option = contract.resolve(member.option_id)
                observed_phenotype = phenotype_identity.identify(
                    thaw_json(option.child_configuration)
                )
                if (
                    option.identity_sha256 != member.option_identity_sha256
                    or observed_phenotype.value_sha256
                    != member.phenotype_identity_sha256
                ):
                    raise RuntimeError("current Heat2D catalog differs from sealed K8")
                child_sha256 = option.child_configuration_sha256
                support_configurations.add(child_sha256)
                source = selected_outcomes.get(member.option_id)
                if source is not None:
                    source_objectives, source_candidate_id = source
                    selected_occurrences += 1
                else:
                    prior = candidate_by_configuration.get(child_sha256)
                    source_objectives = (
                        None if prior is None else _objective_values(prior)
                    )
                    source_candidate_id = (
                        None if prior is None else prior["candidate_id"]
                    )
                    prior_occurrences += int(prior is not None)
                row = {
                    "wave_ordinal": len(waves) + 1,
                    "generation": generation,
                    "parent_slot": audit.get("parent_slot"),
                    "outer_request_sha256": audit["request_sha256"],
                    "slate_allocation_request_sha256": request.request_sha256,
                    "parent_candidate_id": parent_candidate_id,
                    "parent_configuration_sha256": parent_sha256,
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "model_rank": member.model_rank,
                    "option_id": member.option_id,
                    "option_identity_sha256": option.identity_sha256,
                    "configuration_sha256": child_sha256,
                    "configuration": thaw_json(option.child_configuration),
                    "family": member.family,
                    "historical_selected": member.option_id in historical_selected,
                    "structural_posterior_selected": member.option_id
                    in structural_selected,
                    "source_candidate_id": source_candidate_id,
                    "source_objectives": source_objectives,
                }
                ordered.append(row)
                if source_objectives is None:
                    evaluations.append(row)
            waves.append(
                {
                    "wave_ordinal": len(waves) + 1,
                    "generation": generation,
                    "parent_slot": audit.get("parent_slot"),
                    "outer_request_sha256": audit["request_sha256"],
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "slate_allocation_request": request.to_record(),
                    "structural_posterior_decision": decision.to_record(),
                    "historical_selected_option_ids": sorted(historical_selected),
                    "archive_snapshot": archive_receipt,
                    "members": ordered,
                }
            )

    if len(waves) != 6 or sum(len(value["members"]) for value in waves) != 48:
        raise RuntimeError("source must contain six exact K8 waves")
    if selected_occurrences != 24 or prior_occurrences != 0:
        raise RuntimeError("source must expose exactly 24 selected-only outcomes")
    if len(evaluations) != 24:
        raise RuntimeError(
            "Heat K8 completion requires 24 unselected occurrences: "
            f"fresh={len(evaluations)}, unique={len(support_configurations)}"
        )
    return evaluations, waves, {
        "k8_occurrences": 48,
        "unique_k8_configurations": len(support_configurations),
        "source_selected_outcomes": selected_occurrences,
        "source_prior_outcomes": prior_occurrences,
        "unselected_k8_occurrences": len(evaluations),
        "fresh_unique_direct_v3_evaluations": len(
            {value["configuration_sha256"] for value in evaluations}
        ),
        "identical_configuration_reuses": len(evaluations)
        - len({value["configuration_sha256"] for value in evaluations}),
    }


def _source_paths(preregistration: Path) -> tuple[Path, ...]:
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/structural_posterior_slate.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/reward/affine_hypervolume.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/objective_resolution/fixed_grid.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/objective_resolution.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/heat2d_constructive/finite_variation_catalog.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/heat2d_constructive/multiobjective_v1.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/heat2d_constructive/phenotype_identity.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/heat2d_constructive/problem_def.py",
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
        preregistration,
    )


def _criteria(analysis_mode: str) -> dict[str, object]:
    if analysis_mode == DESCRIPTIVE_SUPPORT_MODE:
        return {
            "primary_endpoint": (
                "complete_fixed_parent_k8_outcomes_and_exact_k4_subset_utility"
            ),
            "decision": "descriptive_only_no_policy_promotion",
            "comparators": [
                "recorded_structural_posterior_k4",
                "direct_model_top4",
                "uniform_k4_exact_expectation_over_70_subsets",
                "oracle_k4",
            ],
        }
    if analysis_mode != LEGACY_KILL_TEST_MODE:
        raise ValueError(f"unknown Heat2D K8 analysis mode: {analysis_mode}")
    return {
        "primary_endpoint": (
            "sum_over_six_fixed_parent_waves_of_dimensionless_affine_"
            "hypervolume_gain"
        ),
        "promotion_requires_all": {
            "completed_k8_outcomes": 48,
            "provider_calls": 0,
            "api_key_reads": 0,
            "structural_over_uniform_multiple_at_least": (
                PROMOTION_UNIFORM_MULTIPLE
            ),
            "structural_over_historical_multiple_at_least": (
                PROMOTION_HISTORICAL_MULTIPLE
            ),
            "wins_over_uniform_expectation_at_least": PROMOTION_WAVE_WINS,
        },
        "kill_if": (
            "aggregate structural gain is below either uniform-K4 expectation "
            "or historical model-anchored K4 gain"
        ),
        "otherwise": "inconclusive_hold_no_paid_campaign",
    }


def _validated_source_finalization(
    *,
    source_run: Path,
    analysis_mode: str,
    expected_finalization_sha256: str | None,
    expected_content_sha256: str | None,
) -> dict[str, object]:
    finalization = verify_finalized_run_directory(source_run)
    if finalization.get("status") != "completed_healthy":
        raise RuntimeError("Heat2D K8 completion requires a healthy sealed source")
    expected_finalization = expected_finalization_sha256
    expected_content = expected_content_sha256
    if analysis_mode == LEGACY_KILL_TEST_MODE:
        expected_finalization = expected_finalization or EXPECTED_SOURCE_FINALIZATION
        expected_content = expected_content or EXPECTED_SOURCE_CONTENT
    if (
        expected_finalization is not None
        and finalization.get("finalization_sha256") != expected_finalization
    ):
        raise RuntimeError("Heat2D source finalization differs from expectation")
    if (
        expected_content is not None
        and finalization.get("recursive_content_sha256") != expected_content
    ):
        raise RuntimeError("Heat2D source content differs from expectation")
    return finalization


def _prepare(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    preregistration = args.preregistration.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    finalization = _validated_source_finalization(
        source_run=source_run,
        analysis_mode=args.analysis_mode,
        expected_finalization_sha256=args.expected_source_finalization_sha256,
        expected_content_sha256=args.expected_source_content_sha256,
    )
    events = _campaign_events(source_run)
    evaluation_plan, waves, accounting = _build_precommit(events)
    policy = StructuralPosteriorSlatePolicy()
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "preparing_provider_and_pde_free",
        "diagnostic": "heat2d_generic_campaign_k8_support_completion",
        "analysis_mode": args.analysis_mode,
        "source_run": {
            "path": source_run.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": finalization["finalization_sha256"],
            "recursive_content_sha256": finalization["recursive_content_sha256"],
            "mutated": False,
        },
        "preregistration": {
            "path": preregistration.relative_to(WORKSPACE_ROOT).as_posix(),
            "sha256": _sha256_file(preregistration),
        },
        "policy": policy.to_record(),
        "accounting": accounting,
        "criteria": _criteria(args.analysis_mode),
        "claim_boundary": {
            "fixed_parent_fixed_k8_local_allocator_diagnostic": True,
            "campaign_counterfactual": False,
            "paper_ready_efficacy": False,
            "provider_calls": 0,
            "api_key_reads": 0,
            "pde_evaluations_during_preparation": 0,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "source_identity": source_identity(
            _source_paths(preregistration), relative_to=WORKSPACE_ROOT
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_json_atomic(
        output_dir / "evaluation_plan.json",
        {"schema_version": 1, "accounting": accounting, "rows": evaluation_plan},
    )
    write_json_atomic(
        output_dir / "waves.json",
        {"schema_version": 1, "wave_count": len(waves), "waves": waves},
    )
    precommit = {
        "schema_version": 1,
        "status": "prepared_provider_and_pde_free",
        "prepared_at_utc": _utc_now(),
        "source_run_mutated": False,
        "provider_calls": 0,
        "api_key_reads": 0,
        "pde_evaluations": 0,
        "accounting": accounting,
        "criteria": _criteria(args.analysis_mode),
    }
    write_json_atomic(output_dir / "precommit.json", precommit)
    seal = finalize_run_directory(
        output_dir, status="prepared_provider_and_pde_free"
    )
    print(
        json.dumps(
            {
                **precommit,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _objective_resolution() -> FixedGridObjectiveResolution:
    return FixedGridObjectiveResolution(
        metric_specs=tuple(
            FixedGridMetricSpec(
                metric_id=metric_id,
                decimal_origin=Decimal("0"),
                decimal_quantum=Decimal("0.000000000001"),
                rounding_law=FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN,
            )
            for metric_id in OBJECTIVE_IDS
        )
    )


def _evaluation_record(
    *,
    row: dict[str, Any],
    problem: Heat2DMultiObjectiveV1Problem,
    output_dir: Path,
) -> dict[str, Any]:
    configuration = freeze_json(row["configuration"])
    if typed_json_sha256(configuration) != row["configuration_sha256"]:
        raise RuntimeError("prepared configuration identity changed")
    started = time.perf_counter()
    evaluated = problem.evaluate_detailed(thaw_json(configuration))
    elapsed_s = time.perf_counter() - started
    raw_objectives = tuple(
        (objective.name, float(evaluated.objective_values[objective.name]))
        for objective in problem.objectives
    )
    resolution = resolve_objectives(
        _objective_resolution(),
        ObjectiveResolutionRequest(
            configuration=configuration,
            objectives=problem.objectives,
            raw_objectives=raw_objectives,
        ),
    )
    direct = evaluated.direct_v3
    manifest_path = direct.output_dir / "manifest.json"
    if (
        not manifest_path.is_file()
        or direct.manifest.get("all_checks_pass") is not True
        or direct.manifest.get("full_pde_solve_count") != 1
    ):
        raise RuntimeError("fresh direct-v3 evaluation lacks passing evidence")
    return {
        "schema_version": 1,
        "evaluated_at_utc": _utc_now(),
        "wave_ordinal": row["wave_ordinal"],
        "outer_request_sha256": row["outer_request_sha256"],
        "slate_allocation_request_sha256": row[
            "slate_allocation_request_sha256"
        ],
        "model_rank": row["model_rank"],
        "option_id": row["option_id"],
        "option_identity_sha256": row["option_identity_sha256"],
        "configuration_sha256": row["configuration_sha256"],
        "raw_objectives": {
            metric_id: value for metric_id, value in resolution.raw_objectives
        },
        "decision_objectives": {
            metric_id: value for metric_id, value in resolution.decision_objectives
        },
        "objective_resolution_receipt": resolution.to_record(),
        "direct_v3": {
            "output_dir": direct.output_dir.relative_to(output_dir).as_posix(),
            "manifest_path": manifest_path.relative_to(output_dir).as_posix(),
            "manifest_sha256": _sha256_file(manifest_path),
            "genotype_sha256": direct.genotype_sha256,
            "phenotype_sha256": direct.phenotype_sha256,
            "raw_array_sha256": direct.raw_array_sha256,
            "representation_spec_sha256": direct.representation_spec_sha256,
            "finite_element_volume": direct.finite_element_volume,
            "grayness": direct.grayness,
            "gray_fraction_005_095": direct.gray_fraction_005_095,
            "adapter_elapsed_s": direct.adapter_elapsed_s,
            "evaluator_elapsed_s": direct.evaluator_elapsed_s,
            "elapsed_inside_container_s": direct.elapsed_inside_container_s,
            "queue_wait_s": direct.queue_wait_s,
            "peak_rss_bytes": direct.peak_rss_bytes,
            "all_checks_pass": True,
            "full_pde_solve_count": 1,
        },
        "harness_elapsed_s": elapsed_s,
        "physical_direct_v3_evaluation": True,
        "reused_from_evaluation_ordinal": None,
    }


def _reused_evaluation_record(
    *,
    row: dict[str, Any],
    source: dict[str, Any],
    source_ordinal: int,
) -> dict[str, Any]:
    if source["configuration_sha256"] != row["configuration_sha256"]:
        raise RuntimeError("evaluation reuse requires exact configuration identity")
    return {
        **source,
        "evaluated_at_utc": _utc_now(),
        "wave_ordinal": row["wave_ordinal"],
        "outer_request_sha256": row["outer_request_sha256"],
        "slate_allocation_request_sha256": row[
            "slate_allocation_request_sha256"
        ],
        "model_rank": row["model_rank"],
        "option_id": row["option_id"],
        "option_identity_sha256": row["option_identity_sha256"],
        "harness_elapsed_s": 0.0,
        "physical_direct_v3_evaluation": False,
        "reused_from_evaluation_ordinal": source_ordinal,
    }


def _describe_selection(
    *,
    option_ids: tuple[str, ...],
    outcomes: dict[str, dict[str, float]],
    snapshot: AffineHypervolumeSnapshot2D,
    subset_gains: list[float],
) -> dict[str, Any]:
    augmented = snapshot.augmented_hypervolume(
        tuple(outcomes[option_id] for option_id in option_ids)
    )
    gain = max(0.0, augmented - snapshot.base_hypervolume)
    better = sum(value > gain for value in subset_gains)
    ties = sum(value == gain for value in subset_gains)
    return {
        "option_ids": list(option_ids),
        "augmented_hypervolume": augmented,
        "gain": gain,
        "raw_oriented_gain": gain * snapshot.spec.raw_area_scale,
        "rank_min": better + 1,
        "rank_max": better + ties,
        "strictly_better_than_uniform_fraction": (
            sum(gain > value for value in subset_gains) / len(subset_gains)
        ),
    }


def _analyze(
    *,
    waves: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    analysis_mode: str,
) -> dict[str, Any]:
    fresh = {
        (value["outer_request_sha256"], value["option_id"]): value[
            "decision_objectives"
        ]
        for value in evaluations
    }
    if len(fresh) != 24:
        raise RuntimeError("execution must contain 24 unique fresh outcomes")
    wave_results: list[dict[str, Any]] = []
    for wave in waves:
        snapshot = _affine_snapshot(wave["archive_snapshot"])
        outcomes: dict[str, dict[str, float]] = {}
        members = wave["members"]
        for member in members:
            outcome = member["source_objectives"]
            if outcome is None:
                outcome = fresh[(wave["outer_request_sha256"], member["option_id"])]
            outcomes[member["option_id"]] = outcome
        if len(outcomes) != 8:
            raise RuntimeError("completed Heat wave differs from exact K8")
        subset_rows: list[dict[str, Any]] = []
        for subset in combinations(tuple(outcomes), 4):
            gain = snapshot.joint_gain(
                tuple(outcomes[option_id] for option_id in subset)
            )
            subset_rows.append(
                {
                    "option_ids": list(subset),
                    "gain": gain,
                    "raw_oriented_gain": gain * snapshot.spec.raw_area_scale,
                }
            )
        subset_rows.sort(key=lambda value: (-value["gain"], value["option_ids"]))
        gains = [value["gain"] for value in subset_rows]
        structural = tuple(
            value["option_id"]
            for value in wave["structural_posterior_decision"]["selected"]
        )
        historical = tuple(wave["historical_selected_option_ids"])
        direct = tuple(
            value["option_id"]
            for value in sorted(members, key=lambda value: value["model_rank"])[:4]
        )
        wave_results.append(
            {
                "wave_ordinal": wave["wave_ordinal"],
                "generation": wave["generation"],
                "parent_slot": wave["parent_slot"],
                "outer_request_sha256": wave["outer_request_sha256"],
                "base_hypervolume": snapshot.base_hypervolume,
                "structural_posterior_k4": _describe_selection(
                    option_ids=structural,
                    outcomes=outcomes,
                    snapshot=snapshot,
                    subset_gains=gains,
                ),
                "historical_model_anchored_k4": _describe_selection(
                    option_ids=historical,
                    outcomes=outcomes,
                    snapshot=snapshot,
                    subset_gains=gains,
                ),
                "direct_model_top4": _describe_selection(
                    option_ids=direct,
                    outcomes=outcomes,
                    snapshot=snapshot,
                    subset_gains=gains,
                ),
                "uniform_k4": {
                    "support_size": len(gains),
                    "expected_gain": fmean(gains),
                    "minimum_gain": min(gains),
                    "maximum_gain": max(gains),
                    "zero_gain_fraction": sum(value == 0.0 for value in gains)
                    / len(gains),
                },
                "oracle_k4": subset_rows[0],
                "all_k8_outcomes": [
                    {
                        "model_rank": member["model_rank"],
                        "option_id": member["option_id"],
                        "historical_selected": member["historical_selected"],
                        "structural_posterior_selected": member[
                            "structural_posterior_selected"
                        ],
                        "outcomes": outcomes[member["option_id"]],
                    }
                    for member in members
                ],
            }
        )
    structural = [value["structural_posterior_k4"]["gain"] for value in wave_results]
    historical = [
        value["historical_model_anchored_k4"]["gain"] for value in wave_results
    ]
    direct = [value["direct_model_top4"]["gain"] for value in wave_results]
    uniform = [value["uniform_k4"]["expected_gain"] for value in wave_results]
    oracle = [value["oracle_k4"]["gain"] for value in wave_results]
    structural_sum = sum(structural)
    historical_sum = sum(historical)
    uniform_sum = sum(uniform)
    oracle_sum = sum(oracle)
    wins = sum(left > right for left, right in zip(structural, uniform))
    evidence_complete = len(evaluations) == 24 and all(
        value["direct_v3"]["all_checks_pass"] is True
        and value["direct_v3"]["full_pde_solve_count"] == 1
        for value in evaluations
    )
    if analysis_mode == DESCRIPTIVE_SUPPORT_MODE:
        promotion_conditions = None
        decision = "descriptive_support_completion_no_policy_promotion"
    else:
        if analysis_mode != LEGACY_KILL_TEST_MODE:
            raise RuntimeError(f"unknown sealed Heat2D analysis mode: {analysis_mode}")
        promotion_conditions = {
            "complete_authenticated_k8_support": evidence_complete,
            "at_least_1_10x_uniform": (
                structural_sum >= PROMOTION_UNIFORM_MULTIPLE * uniform_sum
            ),
            "at_least_1_05x_historical": (
                structural_sum >= PROMOTION_HISTORICAL_MULTIPLE * historical_sum
            ),
            "wins_vs_uniform_at_least_4_of_6": wins >= PROMOTION_WAVE_WINS,
        }
        if all(promotion_conditions.values()):
            decision = "promote_v1_to_prospective_paid_campaign"
        elif structural_sum < uniform_sum or structural_sum < historical_sum:
            decision = "kill_v1"
        else:
            decision = "inconclusive_hold_no_paid_campaign"
    aggregate = {
        "wave_count": len(wave_results),
        "structural_posterior_gain_sum": structural_sum,
        "historical_model_anchored_gain_sum": historical_sum,
        "direct_model_top4_gain_sum": sum(direct),
        "uniform_expected_gain_sum": uniform_sum,
        "oracle_gain_sum": oracle_sum,
        "structural_posterior_minus_historical_gain_sum": (
            structural_sum - historical_sum
        ),
        "structural_posterior_minus_uniform_expected_gain_sum": (
            structural_sum - uniform_sum
        ),
        "structural_posterior_multiple_of_historical": (
            None if historical_sum == 0.0 else structural_sum / historical_sum
        ),
        "structural_posterior_multiple_of_uniform": (
            None if uniform_sum == 0.0 else structural_sum / uniform_sum
        ),
        "structural_posterior_fraction_of_oracle": (
            None if oracle_sum == 0.0 else structural_sum / oracle_sum
        ),
        "wins_vs_uniform_expectation": wins,
        "oracle_ties": sum(left == right for left, right in zip(structural, oracle)),
    }
    return {
        "schema_version": 1,
        "analysis_mode": analysis_mode,
        "claim_scope": (
            "precommitted_fixed_parent_fixed_k8_cross_workload_allocator_"
            "diagnostic_not_campaign_counterfactual_or_efficacy"
        ),
        "waves": wave_results,
        "aggregate": aggregate,
        "promotion_conditions": promotion_conditions,
        "preregistered_decision": decision,
    }


def _execute(args: argparse.Namespace) -> int:
    preparation = args.preparation.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    preparation_seal = verify_finalized_run_directory(preparation)
    if preparation_seal.get("status") != "prepared_provider_and_pde_free":
        raise RuntimeError("execution requires the sealed provider/PDE-free precommit")
    prepared_manifest = _read_json(preparation / "manifest.json")
    analysis_mode = prepared_manifest.get("analysis_mode")
    if analysis_mode not in {LEGACY_KILL_TEST_MODE, DESCRIPTIVE_SUPPORT_MODE}:
        raise RuntimeError("sealed Heat2D preparation has no valid analysis mode")
    policy = StructuralPosteriorSlatePolicy()
    if prepared_manifest.get("policy") != policy.to_record():
        raise RuntimeError("current structural-posterior policy differs from precommit")
    source_relative = _object(
        prepared_manifest.get("source_run"), name="prepared source"
    )["path"]
    source_run = (WORKSPACE_ROOT / source_relative).resolve(strict=True)
    source_seal = verify_finalized_run_directory(source_run)
    if (
        source_seal.get("finalization_sha256")
        != prepared_manifest["source_run"]["finalization_sha256"]
        or source_seal.get("recursive_content_sha256")
        != prepared_manifest["source_run"]["recursive_content_sha256"]
    ):
        raise RuntimeError("sealed source differs from preparation")
    preregistration = (
        WORKSPACE_ROOT / prepared_manifest["preregistration"]["path"]
    ).resolve(strict=True)
    if _sha256_file(preregistration) != prepared_manifest["preregistration"]["sha256"]:
        raise RuntimeError("preregistration changed after precommit")
    plan_record = _read_json(preparation / "evaluation_plan.json")
    waves_record = _read_json(preparation / "waves.json")
    plan = [
        _object(value, name="prepared evaluation")
        for value in _array(plan_record.get("rows"), name="evaluation plan")
    ]
    waves = [
        _object(value, name="prepared wave")
        for value in _array(waves_record.get("waves"), name="prepared waves")
    ]
    if len(plan) != 24 or len(waves) != 6:
        raise RuntimeError("sealed preparation differs from preregistered scale")

    output_dir.mkdir(parents=True, exist_ok=False)
    settings = Heat2DDirectV3Settings(
        output_root=output_dir / "pde",
        resolution=1001,
        cpu_set="8",
        timeout_s=180.0,
        required_numpy_version="2.3.5",
        external_concurrency=1,
    )
    problem = Heat2DMultiObjectiveV1Problem(settings)
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "running",
        "diagnostic": "heat2d_generic_campaign_k8_support_completion",
        "analysis_mode": analysis_mode,
        "preparation": {
            "path": preparation.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": preparation_seal["finalization_sha256"],
            "recursive_content_sha256": preparation_seal[
                "recursive_content_sha256"
            ],
        },
        "source_run": prepared_manifest["source_run"],
        "preregistration": prepared_manifest["preregistration"],
        "policy": policy.to_record(),
        "criteria": prepared_manifest["criteria"],
        "workload": {
            "id": "engibench-heatconduction2d-constructive-pareto-v1",
            "evaluator": "direct_v3",
            "resolution": 1001,
            "cpu_set": "8",
            "external_concurrency": 1,
            "unselected_k8_occurrences": len(plan),
            "fresh_unique_direct_v3_evaluations": len(
                {value["configuration_sha256"] for value in plan}
            ),
            "identical_configuration_reuses": len(plan)
            - len({value["configuration_sha256"] for value in plan}),
            "decision_grid_quantum": "0.000000000001",
        },
        "claim_boundary": {
            "provider_calls": 0,
            "api_key_reads": 0,
            "fixed_parent_fixed_k8_local_allocator_diagnostic": True,
            "campaign_counterfactual": False,
            "paper_ready_efficacy": False,
        },
        "source_identity": source_identity(
            _source_paths(preregistration), relative_to=WORKSPACE_ROOT
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    preflight = problem.preflight()
    write_json_atomic(output_dir / "evaluator_preflight.json", preflight)
    evaluation_dir = output_dir / "evaluation_records"
    evaluation_dir.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    evaluations: list[dict[str, Any]] = []
    evaluated_by_configuration: dict[str, tuple[int, dict[str, Any]]] = {}
    for ordinal, row in enumerate(plan, start=1):
        prior = evaluated_by_configuration.get(row["configuration_sha256"])
        if prior is None:
            record = _evaluation_record(
                row=row, problem=problem, output_dir=output_dir
            )
            evaluated_by_configuration[row["configuration_sha256"]] = (
                ordinal,
                record,
            )
        else:
            source_ordinal, source_record = prior
            record = _reused_evaluation_record(
                row=row,
                source=source_record,
                source_ordinal=source_ordinal,
            )
        write_json_atomic(
            evaluation_dir / f"evaluation_{ordinal:02d}.json", record
        )
        evaluations.append(record)
        print(
            json.dumps(
                {
                    "progress": f"{ordinal}/{len(plan)}",
                    "wave_ordinal": record["wave_ordinal"],
                    "option_id": record["option_id"],
                    "thermal_term": record["decision_objectives"][
                        THERMAL_OBJECTIVE_NAME
                    ],
                    "material_fraction": record["decision_objectives"][
                        MATERIAL_OBJECTIVE_NAME
                    ],
                    "elapsed_s": record["harness_elapsed_s"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    wall_s = time.perf_counter() - started
    analysis = _analyze(
        waves=waves,
        evaluations=evaluations,
        analysis_mode=analysis_mode,
    )
    write_json_atomic(output_dir / "allocation_analysis.json", analysis)
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "preregistered_decision": analysis["preregistered_decision"],
        "promotion_conditions": analysis["promotion_conditions"],
        "aggregate": analysis["aggregate"],
        "accounting": {
            "completed_k8_outcomes": 48,
            "source_selected_outcomes": 24,
            "unselected_k8_occurrences": len(evaluations),
            "fresh_unique_direct_v3_evaluations": sum(
                value["physical_direct_v3_evaluation"] is True
                for value in evaluations
            ),
            "identical_configuration_reuses": sum(
                value["physical_direct_v3_evaluation"] is False
                for value in evaluations
            ),
            "passing_unselected_outcome_records": sum(
                value["direct_v3"]["all_checks_pass"] is True
                for value in evaluations
            ),
        },
        "wall_s": wall_s,
        "mean_fresh_evaluation_wall_s": fmean(
            value["harness_elapsed_s"]
            for value in evaluations
            if value["physical_direct_v3_evaluation"] is True
        ),
        "provider_calls": 0,
        "api_key_reads": 0,
        "source_run_mutated": False,
        "claim_scope": analysis["claim_scope"],
    }
    write_json_atomic(output_dir / "result.json", result)
    seal = finalize_run_directory(output_dir, status="completed")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    prepare.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    prepare.add_argument(
        "--analysis-mode",
        choices=(LEGACY_KILL_TEST_MODE, DESCRIPTIVE_SUPPORT_MODE),
        default=LEGACY_KILL_TEST_MODE,
    )
    prepare.add_argument("--expected-source-finalization-sha256")
    prepare.add_argument("--expected-source-content-sha256")
    prepare.add_argument("--output-dir", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--preparation", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "execute":
        return _execute(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
