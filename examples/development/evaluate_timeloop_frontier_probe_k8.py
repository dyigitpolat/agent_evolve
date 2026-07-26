#!/usr/bin/env python3
"""Seal and execute the held-out Timeloop K8 allocation diagnostic.

``prepare`` reconstructs the six exact K8 slates from the sealed DeepSeek G6
campaign, replays the frozen frontier-probe decision, and seals every
recorded-but-unselected child without running Docker or reading a credential.

``execute`` accepts only that sealed plan, evaluates the missing children with
the unchanged pinned Timeloop v2 boundary, completes all 70 K4 subsets per
wave, and adjudicates the preregistered held-out allocator criteria.  This is a
local fixed-parent/fixed-slate diagnostic, not a campaign counterfactual.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
from statistics import fmean
import sys
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (  # noqa: E402
    TypedConfigurationPhenotypeIdentityPolicy,
    bind_finite_variation_catalog,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.domain.ids import ArtifactId  # noqa: E402
from agent_evolve.domain.typed_json import thaw_json  # noqa: E402
from agent_evolve.infrastructure.artifacts.filesystem import (  # noqa: E402
    FileSystemArtifactStore,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineObjectiveAxis,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (  # noqa: E402
    AffineHypervolume3DSpec,
    AffineHypervolumeSnapshot3D,
)
from agent_evolve.policies.selection.calibrated_slate_codec import (  # noqa: E402
    decode_slate_allocation_request_record,
)
from agent_evolve.policies.selection.frontier_probe_slate import (  # noqa: E402
    FrontierProbeSlatePolicy,
)
from agent_evolve.ports.artifact_store import decode_json_bytes  # noqa: E402
from examples.benchmarks.timeloop_codesign.v2.candidate import (  # noqa: E402
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    OBJECTIVE_NAMES,
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2DockerEvaluator,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import (  # noqa: E402
    TimeloopV2FiniteVariationCatalog,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
)
DEFAULT_SOURCE_RUN = (
    ARTIFACT_ROOT / "benchmark_q1/timeloop_codesign/frontier_probe_g6/"
    "deepseek_live_v3_20260719"
)
DEFAULT_POLICY_DEVELOPMENT = (
    ARTIFACT_ROOT
    / "allocator_v2/frontier_probe_three_panel_development_replay_v1_20260719"
)
DEFAULT_PREREGISTRATION = (
    WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts/"
    "258_timeloop_k8_heldout_allocator_preregistration.md"
)

EXPECTED_SOURCE_FINALIZATION = (
    "534da3962b08b91e263dd29774b9e1c92f13b092e43587e6d29b6f312345c7c5"
)
EXPECTED_SOURCE_CONTENT = (
    "9387e27563a7dc5d12b0e98513ea51b36345fb015a44cf2684179e9e9e52def3"
)
EXPECTED_POLICY_DEVELOPMENT_FINALIZATION = (
    "fe7264fb3b567a5cbdbf09487f1a1c1b31482bb615f795191733c5e7bddafa7a"
)
EXPECTED_POLICY_DEVELOPMENT_CONTENT = (
    "d1fd839c865df5002fa7cc4dc5add06d5142ea9e55c536fe9d4a3fc3b5494b7c"
)
CPU_SET = "8"
TIMEOUT_S = 180.0
PORTFOLIO_SIZE = 4
SLATE_SIZE = 8
WAVE_COUNT = 6
UNSELECTED_OCCURRENCES = 24
PROMOTION_UNIFORM_MULTIPLE = 1.10
PROMOTION_ORACLE_FRACTION = 0.90
PROMOTION_WAVE_WINS = 4


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


def _relative(path: Path) -> str:
    return path.resolve(strict=True).relative_to(WORKSPACE_ROOT).as_posix()


def _campaign_events(source_run: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: str | None = None
    for expected_sequence, row in enumerate(
        read_jsonl(source_run / "campaign_events.jsonl"),
        start=1,
    ):
        authenticated = _object(
            row.get("authenticated_record"), name="authenticated event wrapper"
        )
        event = _object(
            authenticated.get("authenticated_campaign_event"),
            name="authenticated campaign event",
        )
        if (
            event.get("sequence") != expected_sequence
            or event.get("previous_event_sha256") != previous
        ):
            raise RuntimeError("source campaign event chain is not contiguous")
        event_sha256 = event.get("event_sha256")
        if type(event_sha256) is not str:
            raise RuntimeError("source campaign event omitted its identity")
        previous = event_sha256
        result.append(event)
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
    start_receipt = _object(start_payload.get("start_receipt"), name="start receipt")
    for seed_value in _array(start_receipt.get("seed_receipts"), name="seed receipts"):
        seed = _object(seed_value, name="seed receipt")
        evidence = _object(seed.get("evidence"), name="seed evidence")
        add(evidence.get("candidate"))
    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _object(event.get("payload"), name="stage payload")
        stage = _object(payload.get("stage_receipt"), name="stage receipt")
        stage_result = _object(stage.get("result"), name="stage result")
        for candidate in _array(stage_result.get("candidates"), name="candidates"):
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
    if set(result) != set(OBJECTIVE_NAMES):
        raise RuntimeError("candidate differs from exact Timeloop objectives")
    if (
        candidate.get("valid") is not True
        or candidate.get("operator_compliant") is not True
        or candidate.get("evidence_compliant") is not True
    ):
        raise RuntimeError("source candidate is not valid and evidence-compliant")
    return result


def _configuration_index(source_run: Path) -> dict[str, dict[str, Any]]:
    """Recover typed configurations only from verified detailed receipts."""

    root = source_run / "artifact_store"
    store = FileSystemArtifactStore(root)
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.artifact")):
        artifact_id = ArtifactId(path.stem)
        reference = store.stat(artifact_id)
        if reference.media_type != "application/json":
            continue
        decoded = decode_json_bytes(
            store.read_bytes(artifact_id, expected_media_type="application/json")
        )
        if type(decoded) is not dict or decoded.get("receipt_kind") != (
            "timeloop_v2_detailed_evaluation"
        ):
            continue
        configuration_value = decoded.get("configuration")
        if type(configuration_value) is not dict:
            continue
        configuration = normalize_candidate(configuration_value).model_dump(
            mode="python"
        )
        frozen = freeze_json(configuration)
        configuration_sha256 = typed_json_sha256(frozen)
        evaluation = _object(decoded.get("evaluation"), name="detailed evaluation")
        if evaluation.get("candidate_sha256") != candidate_sha256(configuration):
            raise RuntimeError("detailed receipt configuration identity drift")
        prior = result.get(configuration_sha256)
        if prior is not None and prior != configuration:
            raise RuntimeError("one configuration identity resolves inconsistently")
        result[configuration_sha256] = configuration
    if not result:
        raise RuntimeError("source artifact store exposes no detailed configurations")
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
    if set(result) != set(OBJECTIVE_NAMES):
        raise RuntimeError("objective point differs from Timeloop metric set")
    return result


def _affine_snapshot(receipt_value: object) -> AffineHypervolumeSnapshot3D:
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
    if len(axes) != 3:
        raise RuntimeError("Timeloop affine snapshot must have exactly three axes")
    spec = AffineHypervolume3DSpec(
        axes=(axes[0], axes[1], axes[2]),
        reference_provenance=spec_record["reference_provenance"],
    )
    if spec.to_record() != spec_record:
        raise RuntimeError("reconstructed affine specification differs from trace")
    snapshot = AffineHypervolumeSnapshot3D.create(
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
    if supplemental.get("decision_sha256") != audit.get(
        "decision_sha256"
    ) or supplemental.get("request_sha256") != audit.get("request_sha256"):
        raise RuntimeError("supplemental audit does not join its selector audit")
    payload = _object(supplemental.get("payload"), name="supplemental payload")
    allocation = _object(payload.get("allocation"), name="historical allocation")
    request = decode_slate_allocation_request_record(allocation.get("source_request"))
    return allocation, request


def _build_precommit(
    *,
    source_run: Path,
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates = _candidate_index(events)
    configurations = _configuration_index(source_run)
    archives = _archive_receipts(events)
    panel = frozen_network_panel("resnet50")
    catalog = TimeloopV2FiniteVariationCatalog(panel)
    phenotype_identity = TypedConfigurationPhenotypeIdentityPolicy()
    policy = FrontierProbeSlatePolicy()
    waves: list[dict[str, Any]] = []
    plan: list[dict[str, Any]] = []
    source_selected = 0
    source_prior = 0

    candidate_by_configuration: dict[str, dict[str, Any]] = {}
    for candidate in candidates.values():
        configuration_sha256 = candidate["configuration_sha256"]
        prior = candidate_by_configuration.get(configuration_sha256)
        if prior is not None and _objective_values(prior) != _objective_values(
            candidate
        ):
            raise RuntimeError("one source configuration has inconsistent outcomes")
        candidate_by_configuration[configuration_sha256] = candidate

    known_phenotype_sha256s = {
        candidate["configuration_sha256"]
        for candidate in candidates.values()
        if candidate.get("generation") == 0
    }
    if len(known_phenotype_sha256s) != 2:
        raise RuntimeError("Timeloop source differs from its two-seed start")

    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _object(event.get("payload"), name="stage payload")
        stage = _object(payload.get("stage_receipt"), name="stage receipt")
        generation = stage.get("generation")
        if type(generation) is not int:
            raise RuntimeError("stage generation must be an exact integer")
        stage_result = _object(stage.get("result"), name="stage result")
        stage_candidates = [
            _object(value, name="stage candidate")
            for value in _array(stage_result.get("candidates"), name="candidates")
        ]
        if generation % 2 == 0:
            known_phenotype_sha256s.update(
                value["configuration_sha256"] for value in stage_candidates
            )
            continue
        known_before_stage = tuple(sorted(known_phenotype_sha256s))
        wave_receipts = {
            value["selection_call_id"]: value
            for value in (
                _object(item, name="portfolio wave receipt")
                for item in _array(
                    stage_result.get("portfolio_wave_receipts"),
                    name="portfolio wave receipts",
                )
            )
        }
        audits = [
            _object(value, name="selector audit")
            for value in _array(stage.get("selector_audits"), name="selector audits")
        ]
        audits.sort(key=lambda value: int(value["parent_slot"]))
        if len(audits) != 2 or len(wave_receipts) != 2:
            raise RuntimeError("each odd generation must contain two lanes")
        archive_receipt = archives.get(generation)
        if archive_receipt is None:
            raise RuntimeError("portfolio generation omitted affine snapshot")

        for audit in audits:
            call_id = audit.get("selector_call_id")
            if type(call_id) is not str or call_id not in wave_receipts:
                raise RuntimeError("selector audit has no exact wave receipt")
            wave_receipt = wave_receipts[call_id]
            plaintext = _object(audit.get("plaintext_audit"), name="plaintext audit")
            frame = _prompt_frame(plaintext.get("request_text"))
            context = _object(frame.get("context"), name="selector context")
            parent_prompt = _object(context.get("parent"), name="prompt parent")
            parent_sha256 = wave_receipt.get("parent_configuration_sha256")
            if (
                type(parent_sha256) is not str
                or parent_prompt.get("configuration_sha256") != parent_sha256
                or parent_sha256 not in configurations
            ):
                raise RuntimeError("prompt and wave receipt differ on parent identity")
            parent_configuration = configurations[parent_sha256]
            frozen_parent = freeze_json(parent_configuration)
            if typed_json_sha256(frozen_parent) != parent_sha256:
                raise RuntimeError("recovered parent configuration identity drift")
            base_contract = bind_finite_variation_catalog(catalog, frozen_parent)
            eligible_view = eligible_finite_variation_view(
                contract=base_contract,
                option_phenotypes=exact_configuration_phenotype_bindings(base_contract),
                known_phenotype_sha256s=known_before_stage,
            )
            contract = eligible_view.contract
            prompt_contract = _object(
                frame.get("finite_variation_contract"), name="prompt contract"
            )
            if (
                prompt_contract.get("parent_configuration_sha256") != parent_sha256
                or prompt_contract.get("contract_identity_sha256")
                != contract.identity_sha256
                or prompt_contract.get("catalog_definition_sha256")
                != contract.catalog_definition_sha256
                or prompt_contract.get("catalog_id") != contract.catalog_id
                or prompt_contract.get("catalog_version") != contract.catalog_version
            ):
                raise RuntimeError("current Timeloop catalog differs from prompt")

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
            if decision.to_record() != allocation:
                raise RuntimeError("frozen policy does not replay source decision")
            primary_selected = {value.option_id for value in decision.selected}
            historical_selected = {
                _object(value, name="historical selection")["option_id"]
                for value in _array(
                    allocation.get("selected"), name="historical selections"
                )
            }
            if len(primary_selected) != PORTFOLIO_SIZE or (
                primary_selected != historical_selected
            ):
                raise RuntimeError("source decision differs from primary policy")

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
            for member in sorted(
                request.slate.members, key=lambda value: value.model_rank
            ):
                option = contract.resolve(member.option_id)
                observed_phenotype = phenotype_identity.identify(
                    thaw_json(option.child_configuration)
                )
                if (
                    option.identity_sha256 != member.option_identity_sha256
                    or observed_phenotype.value_sha256
                    != member.phenotype_identity_sha256
                ):
                    raise RuntimeError(
                        "current Timeloop catalog differs from sealed K8"
                    )
                child_sha256 = option.child_configuration_sha256
                source = selected_outcomes.get(member.option_id)
                source_kind: str | None
                if source is not None:
                    source_objectives, source_candidate_id = source
                    source_kind = "selected_source_outcome"
                    source_selected += 1
                    if (
                        candidates[source_candidate_id]["configuration_sha256"]
                        != child_sha256
                    ):
                        raise RuntimeError("selected outcome names a foreign child")
                else:
                    prior = candidate_by_configuration.get(child_sha256)
                    if prior is None:
                        source_objectives = None
                        source_candidate_id = None
                        source_kind = None
                    else:
                        source_objectives = _objective_values(prior)
                        source_candidate_id = prior["candidate_id"]
                        source_kind = "prior_exact_configuration_outcome"
                        source_prior += 1
                row = {
                    "wave_ordinal": len(waves) + 1,
                    "generation": generation,
                    "parent_slot": audit["parent_slot"],
                    "outer_request_sha256": audit["request_sha256"],
                    "slate_allocation_request_sha256": request.request_sha256,
                    "parent_candidate_id": parent_candidate_id,
                    "parent_configuration_sha256": parent_sha256,
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "model_rank": member.model_rank,
                    "option_id": member.option_id,
                    "option_identity_sha256": option.identity_sha256,
                    "phenotype_identity_sha256": member.phenotype_identity_sha256,
                    "configuration_sha256": child_sha256,
                    "candidate_sha256": candidate_sha256(
                        thaw_json(option.child_configuration)
                    ),
                    "configuration": thaw_json(option.child_configuration),
                    "family": member.family,
                    "locus_key": member.locus_key,
                    "predictions": [
                        {
                            "metric_id": value.metric_id,
                            "asserted_direction": value.asserted_direction.value,
                            "confidence": value.confidence.value,
                        }
                        for value in member.predictions
                    ],
                    "structural_evidence": member.structural_evidence.to_record(),
                    "primary_selected": member.option_id in primary_selected,
                    "source_kind": source_kind,
                    "source_candidate_id": source_candidate_id,
                    "source_objectives": source_objectives,
                }
                ordered.append(row)
                if source_objectives is None:
                    plan.append(row)
            if len(ordered) != SLATE_SIZE:
                raise RuntimeError("reconstructed slate differs from K8")
            waves.append(
                {
                    "wave_ordinal": len(waves) + 1,
                    "generation": generation,
                    "parent_slot": audit["parent_slot"],
                    "outer_request_sha256": audit["request_sha256"],
                    "slate_allocation_request": request.to_record(),
                    "slate_allocation_request_sha256": request.request_sha256,
                    "parent_candidate_id": parent_candidate_id,
                    "parent_configuration_sha256": parent_sha256,
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "archive_snapshot": archive_receipt,
                    "eligibility_receipt": eligible_view.receipt.to_record(),
                    "frontier_probe_decision": decision.to_record(),
                    "primary_selected_option_ids": [
                        value.option_id for value in decision.selected
                    ],
                    "members": ordered,
                }
            )
        known_phenotype_sha256s.update(
            value["configuration_sha256"] for value in stage_candidates
        )

    if (
        len(waves) != WAVE_COUNT
        or source_selected != WAVE_COUNT * PORTFOLIO_SIZE
        or len(plan) + source_prior != UNSELECTED_OCCURRENCES
    ):
        raise RuntimeError("Timeloop source differs from preregistered K8 scale")
    unique_planned = len({value["configuration_sha256"] for value in plan})
    return (
        plan,
        waves,
        {
            "wave_count": len(waves),
            "completed_support_occurrences": WAVE_COUNT * SLATE_SIZE,
            "source_selected_outcomes": source_selected,
            "source_prior_exact_outcomes": source_prior,
            "unselected_k8_occurrences": UNSELECTED_OCCURRENCES,
            "planned_evaluation_occurrences": len(plan),
            "planned_unique_physical_evaluations": unique_planned,
            "planned_identity_reuses": len(plan) - unique_planned,
        },
    )


def _criteria() -> dict[str, object]:
    return {
        "primary_endpoint": ("sum_of_fixed_parent_fixed_k8_affine_3d_hypervolume_gain"),
        "promotion_requires_all": {
            "complete_authenticated_terminal_support": True,
            "runtime_system_failures": 0,
            "provider_calls": 0,
            "api_key_reads": 0,
            "primary_over_uniform_multiple_at_least": (PROMOTION_UNIFORM_MULTIPLE),
            "primary_fraction_of_oracle_at_least": PROMOTION_ORACLE_FRACTION,
            "wins_over_uniform_expectation_at_least": PROMOTION_WAVE_WINS,
            "wave_count": WAVE_COUNT,
        },
        "kill_if": "aggregate_primary_gain_below_uniform_expectation",
        "otherwise": "inconclusive_hold_no_paid_campaign",
    }


def _source_paths(preregistration: Path) -> tuple[Path, ...]:
    return (
        Path(__file__),
        preregistration,
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/frontier_probe_slate.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/selection/calibrated_slate_codec.py",
        AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/reward/affine_hypervolume_3d.py",
        AGENT_EVOLVE_ROOT
        / "examples/benchmarks/timeloop_codesign/v2/finite_variation_catalog.py",
        AGENT_EVOLVE_ROOT / "examples/benchmarks/timeloop_codesign/v2/evaluator.py",
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
    )


def _prepare(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    policy_development = args.policy_development.expanduser().resolve(strict=True)
    preregistration = args.preregistration.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_seal = verify_finalized_run_directory(source_run)
    development_seal = verify_finalized_run_directory(policy_development)
    if (
        source_seal.get("status") != "completed_healthy"
        or source_seal.get("finalization_sha256") != EXPECTED_SOURCE_FINALIZATION
        or source_seal.get("recursive_content_sha256") != EXPECTED_SOURCE_CONTENT
    ):
        raise RuntimeError("Timeloop source differs from held-out preregistration")
    if (
        development_seal.get("finalization_sha256")
        != EXPECTED_POLICY_DEVELOPMENT_FINALIZATION
        or development_seal.get("recursive_content_sha256")
        != EXPECTED_POLICY_DEVELOPMENT_CONTENT
    ):
        raise RuntimeError("frontier-probe development evidence changed")
    events = _campaign_events(source_run)
    plan, waves, accounting = _build_precommit(
        source_run=source_run,
        events=events,
    )
    policy = FrontierProbeSlatePolicy()
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "preparing_provider_and_timeloop_free",
        "diagnostic": "heldout_timeloop_frontier_probe_k8",
        "source_run": {
            "path": _relative(source_run),
            "finalization_sha256": source_seal["finalization_sha256"],
            "recursive_content_sha256": source_seal["recursive_content_sha256"],
            "mutated": False,
        },
        "policy_development": {
            "path": _relative(policy_development),
            "finalization_sha256": development_seal["finalization_sha256"],
            "recursive_content_sha256": development_seal["recursive_content_sha256"],
            "timeloop_outcomes_used": False,
        },
        "preregistration": {
            "path": _relative(preregistration),
            "sha256": _sha256_file(preregistration),
        },
        "policy": policy.to_record(),
        "accounting": accounting,
        "criteria": _criteria(),
        "claim_boundary": {
            "fixed_parent_fixed_k8_local_allocator_diagnostic": True,
            "campaign_counterfactual": False,
            "paper_ready_efficacy": False,
            "provider_calls": 0,
            "api_key_reads": 0,
            "timeloop_evaluations_during_preparation": 0,
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
        {"schema_version": 1, "accounting": accounting, "rows": plan},
    )
    write_json_atomic(
        output_dir / "waves.json",
        {"schema_version": 1, "wave_count": len(waves), "waves": waves},
    )
    precommit = {
        "schema_version": 1,
        "status": "prepared_provider_and_timeloop_free",
        "prepared_at_utc": _utc_now(),
        "source_run_mutated": False,
        "provider_calls": 0,
        "api_key_reads": 0,
        "timeloop_evaluations": 0,
        "accounting": accounting,
        "criteria": _criteria(),
    }
    write_json_atomic(output_dir / "precommit.json", precommit)
    seal = finalize_run_directory(
        output_dir, status="prepared_provider_and_timeloop_free"
    )
    print(
        json.dumps(
            {
                **precommit,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
                "recursive_content_sha256": seal["recursive_content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evidence_file_record(path: Path, *, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Timeloop evaluation omitted evidence file: {path.name}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _evaluate(
    *,
    row: dict[str, Any],
    evaluator: TimeloopV2DockerEvaluator,
    output_dir: Path,
) -> dict[str, Any]:
    configuration = normalize_candidate(row["configuration"]).model_dump(mode="python")
    frozen = freeze_json(configuration)
    if (
        typed_json_sha256(frozen) != row["configuration_sha256"]
        or candidate_sha256(configuration) != row["candidate_sha256"]
    ):
        raise RuntimeError("prepared Timeloop candidate identity changed")
    started = time.perf_counter()
    try:
        result = evaluator.evaluate(configuration)
    except TimeloopV2CandidateInfeasibleError as error:
        observation = error.observation
        status = "candidate_infeasible"
        objectives = None
    else:
        observation = result
        status = "passed"
        objectives = dict(result.objective_values)
    harness_elapsed_s = time.perf_counter() - started
    if observation.candidate_sha256 != row["candidate_sha256"]:
        raise RuntimeError("evaluator returned a foreign Timeloop candidate")
    call_dir = observation.output_dir.resolve(strict=True)
    evidence = [
        _evidence_file_record(call_dir / name, root=output_dir)
        for name in ("evaluation-bundle.json", "result.json", "host_receipt.json")
    ]
    return {
        "schema_version": 1,
        "evaluated_at_utc": _utc_now(),
        "wave_ordinal": row["wave_ordinal"],
        "generation": row["generation"],
        "parent_slot": row["parent_slot"],
        "outer_request_sha256": row["outer_request_sha256"],
        "slate_allocation_request_sha256": row["slate_allocation_request_sha256"],
        "model_rank": row["model_rank"],
        "option_id": row["option_id"],
        "option_identity_sha256": row["option_identity_sha256"],
        "configuration_sha256": row["configuration_sha256"],
        "candidate_sha256": row["candidate_sha256"],
        "status": status,
        "objectives": objectives,
        "compiled_plan_sha256": observation.compiled_plan_sha256,
        "panel_sha256": observation.panel_sha256,
        "incomplete_medoid_ordinals": list(
            getattr(observation, "incomplete_medoid_ordinals", ())
        ),
        "evaluator_elapsed_s": observation.evaluator_elapsed_s,
        "queue_wait_s": observation.queue_wait_s,
        "harness_elapsed_s": harness_elapsed_s,
        "output_dir": call_dir.relative_to(output_dir).as_posix(),
        "evidence_files": evidence,
        "physical_timeloop_evaluation": True,
        "reused_from_evaluation_ordinal": None,
    }


def _reuse_evaluation(
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
        "generation": row["generation"],
        "parent_slot": row["parent_slot"],
        "outer_request_sha256": row["outer_request_sha256"],
        "slate_allocation_request_sha256": row["slate_allocation_request_sha256"],
        "model_rank": row["model_rank"],
        "option_id": row["option_id"],
        "option_identity_sha256": row["option_identity_sha256"],
        "harness_elapsed_s": 0.0,
        "physical_timeloop_evaluation": False,
        "reused_from_evaluation_ordinal": source_ordinal,
    }


def _selection(
    *,
    option_ids: tuple[str, ...],
    outcomes: dict[str, dict[str, float] | None],
    snapshot: AffineHypervolumeSnapshot3D,
    subset_gains: list[float],
) -> dict[str, Any]:
    points = tuple(
        value for option_id in option_ids if (value := outcomes[option_id]) is not None
    )
    gain = snapshot.joint_gain(points)
    better = sum(value > gain for value in subset_gains)
    ties = sum(value == gain for value in subset_gains)
    return {
        "option_ids": list(option_ids),
        "successful_point_count": len(points),
        "gain": gain,
        "raw_oriented_gain": gain * snapshot.spec.raw_volume_scale,
        "augmented_hypervolume": snapshot.base_hypervolume + gain,
        "rank_min": better + 1,
        "rank_max": better + ties,
        "strictly_better_than_uniform_fraction": (
            sum(gain > value for value in subset_gains) / len(subset_gains)
        ),
    }


def _actual_direction(child: float, parent: float) -> str:
    if child < parent:
        return "decrease"
    if child > parent:
        return "increase"
    return "unchanged"


def _parent_relation(
    child: dict[str, float] | None,
    parent: dict[str, float],
) -> str:
    if child is None:
        return "candidate_infeasible"
    lower = [child[key] < parent[key] for key in OBJECTIVE_NAMES]
    higher = [child[key] > parent[key] for key in OBJECTIVE_NAMES]
    if not any(lower) and not any(higher):
        return "equivalent"
    if any(lower) and not any(higher):
        return "strictly_dominates_parent"
    if any(higher) and not any(lower):
        return "strictly_dominated_by_parent"
    return "incomparable"


def _analyze(
    *,
    waves: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluated = {
        (value["outer_request_sha256"], value["option_id"]): value
        for value in evaluations
    }
    if len(evaluated) != len(evaluations):
        raise RuntimeError("evaluation occurrence identities are not unique")
    wave_results: list[dict[str, Any]] = []
    rank_rows: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in range(1, SLATE_SIZE + 1)
    }
    overall_known = 0
    overall_correct = 0
    overall_total = 0

    for wave in waves:
        snapshot = _affine_snapshot(wave["archive_snapshot"])
        outcomes: dict[str, dict[str, float] | None] = {}
        statuses: dict[str, str] = {}
        members = [_object(value, name="K8 member") for value in wave["members"]]
        for member in members:
            source = member.get("source_objectives")
            if source is not None:
                outcome = {
                    key: float(value)
                    for key, value in _object(source, name="source objectives").items()
                }
                status = "passed_source"
            else:
                record = evaluated.get(
                    (wave["outer_request_sha256"], member["option_id"])
                )
                if record is None:
                    raise RuntimeError("completed diagnostic omitted an outcome")
                status = str(record["status"])
                outcome_value = record.get("objectives")
                outcome = (
                    None
                    if outcome_value is None
                    else {
                        key: float(value)
                        for key, value in _object(
                            outcome_value, name="evaluation objectives"
                        ).items()
                    }
                )
            if outcome is not None and set(outcome) != set(OBJECTIVE_NAMES):
                raise RuntimeError("completed outcome differs from Timeloop metrics")
            outcomes[member["option_id"]] = outcome
            statuses[member["option_id"]] = status
        if len(outcomes) != SLATE_SIZE:
            raise RuntimeError("completed Timeloop wave differs from exact K8")

        option_order = tuple(member["option_id"] for member in members)
        subset_rows: list[dict[str, Any]] = []
        for subset in combinations(option_order, PORTFOLIO_SIZE):
            points = tuple(
                value
                for option_id in subset
                if (value := outcomes[option_id]) is not None
            )
            subset_rows.append(
                {
                    "option_ids": list(subset),
                    "successful_point_count": len(points),
                    "gain": snapshot.joint_gain(points),
                }
            )
        subset_rows.sort(key=lambda value: (-value["gain"], value["option_ids"]))
        gains = [float(value["gain"]) for value in subset_rows]
        primary = tuple(wave["primary_selected_option_ids"])
        direct = tuple(member["option_id"] for member in members[:PORTFOLIO_SIZE])
        if set(primary) != set(direct):
            raise RuntimeError("held-out Timeloop policy is not the frozen top4 path")

        member_results: list[dict[str, Any]] = []
        canonical_oracle = set(subset_rows[0]["option_ids"])
        for member in members:
            option_id = member["option_id"]
            outcome = outcomes[option_id]
            parent = {
                key: float(value)
                for key, value in _object(
                    member["parent_objectives"], name="parent objectives"
                ).items()
            }
            forecast_total = len(member["predictions"])
            forecast_known = 0
            forecast_correct = 0
            prediction_results: list[dict[str, Any]] = []
            for prediction_value in member["predictions"]:
                prediction = _object(prediction_value, name="prediction")
                asserted = str(prediction["asserted_direction"])
                actual = (
                    None
                    if outcome is None
                    else _actual_direction(
                        outcome[prediction["metric_id"]],
                        parent[prediction["metric_id"]],
                    )
                )
                known = asserted != "unknown" and actual is not None
                correct = known and asserted == actual
                forecast_known += int(known)
                forecast_correct += int(correct)
                prediction_results.append(
                    {
                        **prediction,
                        "actual_direction": actual,
                        "known_for_accuracy": known,
                        "correct": correct,
                    }
                )
            marginal_gain = 0.0 if outcome is None else snapshot.marginal_gain(outcome)
            member_result = {
                "model_rank": member["model_rank"],
                "option_id": option_id,
                "family": member["family"],
                "locus_key": member["locus_key"],
                "primary_selected": member["primary_selected"],
                "status": statuses[option_id],
                "outcomes": outcome,
                "marginal_gain": marginal_gain,
                "positive_marginal_gain": marginal_gain > 0.0,
                "parent_relation": _parent_relation(outcome, parent),
                "canonical_oracle_selected": option_id in canonical_oracle,
                "forecast_total": forecast_total,
                "forecast_known": forecast_known,
                "forecast_correct": forecast_correct,
                "forecast_accuracy": (
                    None if forecast_known == 0 else forecast_correct / forecast_known
                ),
                "predictions": prediction_results,
            }
            member_results.append(member_result)
            rank_rows[int(member["model_rank"])].append(member_result)
            overall_total += forecast_total
            overall_known += forecast_known
            overall_correct += forecast_correct

        wave_results.append(
            {
                "wave_ordinal": wave["wave_ordinal"],
                "generation": wave["generation"],
                "parent_slot": wave["parent_slot"],
                "outer_request_sha256": wave["outer_request_sha256"],
                "base_hypervolume": snapshot.base_hypervolume,
                "frontier_probe_k4": _selection(
                    option_ids=primary,
                    outcomes=outcomes,
                    snapshot=snapshot,
                    subset_gains=gains,
                ),
                "direct_model_top4": _selection(
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
                "oracle_tie_count": sum(value == gains[0] for value in gains),
                "members": member_results,
            }
        )

    primary = [value["frontier_probe_k4"]["gain"] for value in wave_results]
    uniform = [value["uniform_k4"]["expected_gain"] for value in wave_results]
    oracle = [value["oracle_k4"]["gain"] for value in wave_results]
    primary_sum = sum(primary)
    uniform_sum = sum(uniform)
    oracle_sum = sum(oracle)
    wins = sum(left > right for left, right in zip(primary, uniform, strict=True))
    terminal = all(
        value["status"] in {"passed", "candidate_infeasible"} for value in evaluations
    )
    evidence_complete = (
        len(evaluations)
        + sum(
            member["source_objectives"] is not None
            for wave in waves
            for member in wave["members"]
        )
        == WAVE_COUNT * SLATE_SIZE
        and terminal
    )
    promotion_conditions = {
        "complete_authenticated_terminal_support": evidence_complete,
        "runtime_system_failures_zero": terminal,
        "provider_and_api_key_activity_zero": True,
        "at_least_1_10x_uniform": (
            primary_sum >= PROMOTION_UNIFORM_MULTIPLE * uniform_sum
        ),
        "at_least_90_percent_oracle": (
            primary_sum >= PROMOTION_ORACLE_FRACTION * oracle_sum
        ),
        "wins_vs_uniform_at_least_4_of_6": wins >= PROMOTION_WAVE_WINS,
    }
    if all(promotion_conditions.values()):
        decision = "promote_to_prospective_matched_multiseed_campaign"
    elif primary_sum < uniform_sum:
        decision = "kill_frontier_probe_v1"
    else:
        decision = "inconclusive_hold_no_paid_campaign"

    rank_analysis: list[dict[str, Any]] = []
    for rank, rows in rank_rows.items():
        known = sum(int(value["forecast_known"]) for value in rows)
        correct = sum(int(value["forecast_correct"]) for value in rows)
        rank_analysis.append(
            {
                "model_rank": rank,
                "occurrences": len(rows),
                "passed": sum(value["outcomes"] is not None for value in rows),
                "candidate_infeasible": sum(
                    value["outcomes"] is None for value in rows
                ),
                "marginal_gain_sum": sum(value["marginal_gain"] for value in rows),
                "mean_marginal_gain": fmean(
                    float(value["marginal_gain"]) for value in rows
                ),
                "positive_marginal_gain_count": sum(
                    value["positive_marginal_gain"] for value in rows
                ),
                "strict_parent_dominator_count": sum(
                    value["parent_relation"] == "strictly_dominates_parent"
                    for value in rows
                ),
                "canonical_oracle_selection_count": sum(
                    value["canonical_oracle_selected"] for value in rows
                ),
                "forecast_known": known,
                "forecast_correct": correct,
                "forecast_direction_accuracy": (
                    None if known == 0 else correct / known
                ),
            }
        )

    return {
        "schema_version": 1,
        "claim_scope": (
            "preregistered_heldout_fixed_parent_fixed_k8_allocator_diagnostic_"
            "not_campaign_counterfactual_or_efficacy"
        ),
        "waves": wave_results,
        "rank_analysis": rank_analysis,
        "forecast_overall": {
            "total": overall_total,
            "known": overall_known,
            "correct": overall_correct,
            "exact_direction_accuracy": (
                None if overall_known == 0 else overall_correct / overall_known
            ),
        },
        "aggregate": {
            "wave_count": len(wave_results),
            "frontier_probe_gain_sum": primary_sum,
            "direct_model_top4_gain_sum": sum(
                value["direct_model_top4"]["gain"] for value in wave_results
            ),
            "uniform_expected_gain_sum": uniform_sum,
            "oracle_gain_sum": oracle_sum,
            "frontier_probe_minus_uniform_expected_gain_sum": (
                primary_sum - uniform_sum
            ),
            "frontier_probe_multiple_of_uniform": (
                None if uniform_sum == 0.0 else primary_sum / uniform_sum
            ),
            "frontier_probe_fraction_of_oracle": (
                None if oracle_sum == 0.0 else primary_sum / oracle_sum
            ),
            "wins_vs_uniform_expectation": wins,
            "oracle_ties": sum(
                left == right for left, right in zip(primary, oracle, strict=True)
            ),
        },
        "promotion_conditions": promotion_conditions,
        "preregistered_decision": decision,
    }


def _execute(args: argparse.Namespace) -> int:
    preparation = args.preparation.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    preparation_seal = verify_finalized_run_directory(preparation)
    if preparation_seal.get("status") != "prepared_provider_and_timeloop_free":
        raise RuntimeError("execution requires the sealed provider/Timeloop-free plan")
    prepared_manifest = _read_json(preparation / "manifest.json")
    policy = FrontierProbeSlatePolicy()
    if prepared_manifest.get("policy") != policy.to_record():
        raise RuntimeError("current frontier-probe policy differs from precommit")
    source_run = (
        WORKSPACE_ROOT
        / _object(prepared_manifest.get("source_run"), name="prepared source")["path"]
    ).resolve(strict=True)
    source_seal_before = verify_finalized_run_directory(source_run)
    if (
        source_seal_before.get("finalization_sha256")
        != prepared_manifest["source_run"]["finalization_sha256"]
        or source_seal_before.get("recursive_content_sha256")
        != prepared_manifest["source_run"]["recursive_content_sha256"]
    ):
        raise RuntimeError("sealed Timeloop source differs from preparation")
    policy_development = (
        WORKSPACE_ROOT / prepared_manifest["policy_development"]["path"]
    ).resolve(strict=True)
    development_seal = verify_finalized_run_directory(policy_development)
    if (
        development_seal.get("finalization_sha256")
        != prepared_manifest["policy_development"]["finalization_sha256"]
        or development_seal.get("recursive_content_sha256")
        != prepared_manifest["policy_development"]["recursive_content_sha256"]
    ):
        raise RuntimeError("allocator development evidence changed after precommit")
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
    if len(plan) > UNSELECTED_OCCURRENCES or len(waves) != WAVE_COUNT:
        raise RuntimeError("sealed preparation differs from preregistered scale")

    output_dir.mkdir(parents=True, exist_ok=False)
    settings = TimeloopV2Settings(
        output_root=output_dir / "evaluator_calls",
        cpu_set=CPU_SET,
        timeout_s=TIMEOUT_S,
        external_concurrency=1,
    )
    panel = frozen_network_panel("resnet50")
    evaluator = TimeloopV2DockerEvaluator(settings, panel)
    docker_preflight = evaluator.preflight()
    write_json_atomic(output_dir / "docker_preflight.json", docker_preflight)
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "running",
        "diagnostic": "heldout_timeloop_frontier_probe_k8",
        "preparation": {
            "path": _relative(preparation),
            "finalization_sha256": preparation_seal["finalization_sha256"],
            "recursive_content_sha256": preparation_seal["recursive_content_sha256"],
        },
        "source_run": prepared_manifest["source_run"],
        "policy_development": prepared_manifest["policy_development"],
        "preregistration": prepared_manifest["preregistration"],
        "policy": policy.to_record(),
        "criteria": prepared_manifest["criteria"],
        "workload": {
            "id": "timeloop-v2-resnet50-three-medoid-codesign",
            "cpu_set": CPU_SET,
            "external_concurrency": 1,
            "timeout_s": TIMEOUT_S,
            "planned_evaluation_occurrences": len(plan),
            "planned_unique_physical_evaluations": len(
                {value["configuration_sha256"] for value in plan}
            ),
        },
        "docker_preflight": docker_preflight,
        "claim_boundary": {
            "provider_calls": 0,
            "api_key_reads": 0,
            "fixed_parent_fixed_k8_local_allocator_diagnostic": True,
            "campaign_counterfactual": False,
            "paper_ready_efficacy": False,
        },
    }
    write_json_atomic(output_dir / "manifest.json", manifest)

    started = time.perf_counter()
    evaluations: list[dict[str, Any]] = []
    by_configuration: dict[str, tuple[int, dict[str, Any]]] = {}
    try:
        with DurableJsonlJournal(output_dir / "evaluations.jsonl") as journal:
            for ordinal, row in enumerate(plan, start=1):
                configuration_sha256 = row["configuration_sha256"]
                prior = by_configuration.get(configuration_sha256)
                if prior is None:
                    record = _evaluate(
                        row=row,
                        evaluator=evaluator,
                        output_dir=output_dir,
                    )
                    by_configuration[configuration_sha256] = (ordinal, record)
                else:
                    source_ordinal, source = prior
                    record = _reuse_evaluation(
                        row=row,
                        source=source,
                        source_ordinal=source_ordinal,
                    )
                evaluations.append(record)
                journal.append(record)
                print(
                    json.dumps(
                        {
                            "progress": f"{ordinal}/{len(plan)}",
                            "wave_ordinal": row["wave_ordinal"],
                            "model_rank": row["model_rank"],
                            "option_id": row["option_id"],
                            "status": record["status"],
                            "physical": record["physical_timeloop_evaluation"],
                            "harness_elapsed_s": record["harness_elapsed_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException as error:
        write_json_atomic(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "failed_at_utc": _utc_now(),
                "failure_type": type(error).__qualname__,
                "failure_sha256": hashlib.sha256(
                    f"{type(error).__qualname__}\x00{error}".encode("utf-8")
                ).hexdigest(),
                "completed_evaluation_occurrences": len(evaluations),
                "provider_calls": 0,
                "api_key_reads": 0,
            },
        )
        manifest["status"] = "failed"
        write_json_atomic(output_dir / "manifest.json", manifest)
        finalize_run_directory(output_dir, status="failed")
        raise

    analysis = _analyze(waves=waves, evaluations=evaluations)
    wall_s = time.perf_counter() - started
    write_json_atomic(output_dir / "allocation_analysis.json", analysis)
    physical = [value for value in evaluations if value["physical_timeloop_evaluation"]]
    source_seal_after = verify_finalized_run_directory(source_run)
    if source_seal_after != source_seal_before:
        raise RuntimeError("source run changed during held-out execution")
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "source_run_mutated": False,
        "accounting": {
            **prepared_manifest["accounting"],
            "completed_evaluation_occurrences": len(evaluations),
            "physical_timeloop_evaluations": len(physical),
            "identity_reuses": len(evaluations) - len(physical),
            "passed_unselected_occurrences": sum(
                value["status"] == "passed" for value in evaluations
            ),
            "candidate_infeasible_unselected_occurrences": sum(
                value["status"] == "candidate_infeasible" for value in evaluations
            ),
            "runtime_system_failures": 0,
        },
        "wall_s": wall_s,
        "mean_physical_evaluation_wall_s": (
            None
            if not physical
            else fmean(float(value["harness_elapsed_s"]) for value in physical)
        ),
        "aggregate": analysis["aggregate"],
        "forecast_overall": analysis["forecast_overall"],
        "promotion_conditions": analysis["promotion_conditions"],
        "preregistered_decision": analysis["preregistered_decision"],
        "provider_calls": 0,
        "api_key_reads": 0,
        "claim_scope": analysis["claim_scope"],
    }
    write_json_atomic(output_dir / "result.json", result)
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = result["completed_at_utc"]
    manifest["result_sha256"] = _sha256_file(output_dir / "result.json")
    write_json_atomic(output_dir / "manifest.json", manifest)
    seal = finalize_run_directory(output_dir, status="completed")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": seal["finalization_sha256"],
                "recursive_content_sha256": seal["recursive_content_sha256"],
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
        "--policy-development", type=Path, default=DEFAULT_POLICY_DEVELOPMENT
    )
    prepare.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
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
