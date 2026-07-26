#!/usr/bin/env python3
"""Complete every recorded K8 slate from a sealed generic BOiLS campaign.

This is a post-hoc, fixed-parent mechanism diagnostic.  It authenticates the
sealed source directory, reconstructs each model-proposed finite option from
the exact prompt-bound parent, reuses the four recorded selected outcomes, and
evaluates only proposals whose configurations have no source-run outcome.  It
does not call a provider, mutate the source run, update memory, or replay the
campaign trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.typed_json import freeze_json, thaw_json  # noqa: E402
from agent_evolve.campaign_variation_topology import (  # noqa: E402
    CampaignVariationTopology,
    CampaignVariationTopologyMode,
)
from agent_evolve.policies.reward.frozen_wave_archive import (  # noqa: E402
    FrozenArchiveWaveSnapshot2D,
)
from agent_evolve.ports.variation_catalog import (  # noqa: E402
    bind_finite_variation_catalog,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    finite_option_ids_have_pairwise_disjoint_parent_patch_subset,
)
from examples.benchmarks.boils_abc.actions import config_sha256  # noqa: E402
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    BoilsFiniteVariationCatalog,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development.evaluate_boils_calibrated_slate_counterfactual import (  # noqa: E402
    OBJECTIVES,
    _allocation_analysis,
    _evaluate_rejected,
    _parse_affinity_sets,
    _sha256_file,
    _snapshot,
)


DEFAULT_SOURCE_RUN = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/generic_campaign"
    / "boils_generic_g6_deepseek_live_unmatched_dev_r8_20260717"
)
RAW_HYPERVOLUME_SCALE = 80.0 * 12000.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_object(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RuntimeError(f"{name} must be one exact JSON object")
    return value


def _require_array(value: object, name: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{name} must be one exact JSON array")
    return value


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _variation_topology(source_run: Path) -> CampaignVariationTopology:
    """Recover the exact workload-neutral topology sealed by the campaign."""

    manifest = _require_object(
        _read_json(source_run / "manifest.json"), "source campaign manifest"
    )
    workload = _require_object(manifest.get("workload"), "manifest workload")
    record = _require_object(
        workload.get("variation_topology"), "manifest variation topology"
    )
    expected_keys = {
        "schema_version",
        "mode",
        "max_composite_options",
        "required_composite_proposals",
        "selection_exposure",
        "provider_materialization_authority",
        "outcomes_consulted",
    }
    if set(record) != expected_keys or record.get("schema_version") != 1:
        raise RuntimeError("source variation topology record is malformed")
    if (
        record.get("provider_materialization_authority") is not False
        or record.get("outcomes_consulted") is not False
    ):
        raise RuntimeError("source topology violates diagnostic epistemic bounds")
    try:
        topology = CampaignVariationTopology(
            mode=CampaignVariationTopologyMode(record["mode"]),
            max_composite_options=record["max_composite_options"],
            required_composite_proposals=record["required_composite_proposals"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("source variation topology is not reconstructible") from exc
    if topology.to_record() != record:
        raise RuntimeError("reconstructed variation topology differs from manifest")
    return topology


def _objective_values(candidate: dict[str, Any]) -> dict[str, float]:
    rows = _require_array(candidate.get("objectives"), "candidate objectives")
    result: dict[str, float] = {}
    for row in rows:
        cell = _require_object(row, "candidate objective row")
        metric_id = cell.get("metric_id")
        value_hex = cell.get("value_hex")
        if type(metric_id) is not str or type(value_hex) is not str:
            raise RuntimeError("candidate objective row is malformed")
        result[metric_id] = float.fromhex(value_hex)
    if set(result) != {"total_levels", "total_lut_count"}:
        raise RuntimeError("candidate does not carry the exact BOiLS objectives")
    return result


def _campaign_events(source_run: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in read_jsonl(source_run / "campaign_events.jsonl"):
        wrapper = _require_object(row, "campaign event wrapper")
        event = _require_object(
            wrapper.get("authenticated_campaign_event"),
            "authenticated campaign event",
        )
        events.append(event)
    if not events:
        raise RuntimeError("source campaign contains no authenticated events")
    return events


def _candidate_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def add(candidate: object) -> None:
        value = _require_object(candidate, "candidate")
        candidate_id = value.get("candidate_id")
        if type(candidate_id) is not str or not candidate_id:
            raise RuntimeError("candidate has no exact ID")
        prior = result.get(candidate_id)
        if prior is not None and prior != value:
            raise RuntimeError("candidate ID resolves to inconsistent records")
        result[candidate_id] = value

    start = [value for value in events if value.get("kind") == "execution_started"]
    if len(start) != 1:
        raise RuntimeError("source campaign must have one execution_started event")
    start_payload = _require_object(start[0].get("payload"), "start payload")
    start_receipt = _require_object(
        start_payload.get("start_receipt"), "start receipt"
    )
    for seed in _require_array(start_receipt.get("seed_receipts"), "seed receipts"):
        evidence = _require_object(
            _require_object(seed, "seed receipt").get("evidence"),
            "seed evidence",
        )
        add(evidence.get("candidate"))
    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _require_object(event.get("payload"), "stage payload")
        stage = _require_object(payload.get("stage_receipt"), "stage receipt")
        stage_result = _require_object(stage.get("result"), "stage result")
        for candidate in _require_array(
            stage_result.get("candidates"), "stage candidates"
        ):
            add(candidate)
    return result


def _prompt_frame(request_text: str) -> dict[str, Any]:
    if type(request_text) is not str or "{" not in request_text:
        raise RuntimeError("selector request omitted its exact JSON frame")
    suffix = request_text[request_text.index("{") :]
    value, end = json.JSONDecoder().raw_decode(suffix)
    if not suffix[end:].startswith("\n"):
        raise RuntimeError("selector request has a foreign prompt-frame boundary")
    return _require_object(value, "selector prompt frame")


def _archive_snapshot_record(event: dict[str, Any]) -> dict[str, Any]:
    payload = _require_object(event.get("payload"), "archive utility payload")
    utility = _require_object(payload.get("archive_utility"), "archive utility")
    receipt = _require_object(utility.get("snapshot_receipt"), "snapshot receipt")
    spec = _require_object(receipt.get("spec"), "hypervolume specification")
    axes = _require_array(spec.get("axes"), "hypervolume axes")
    reference: dict[str, float] = {}
    reference_record: list[list[str]] = []
    for axis_value in axes:
        axis = _require_object(axis_value, "hypervolume axis")
        metric_id = axis.get("metric_id")
        reference_hex = axis.get("reference_hex")
        if type(metric_id) is not str or type(reference_hex) is not str:
            raise RuntimeError("hypervolume axis is malformed")
        reference[metric_id] = float.fromhex(reference_hex)
        reference_record.append([metric_id, reference_hex])
    raw_points = _require_array(
        receipt.get("raw_archive_points"), "raw archive points"
    )
    point_values: list[dict[str, float]] = []
    for point_value in raw_points:
        point: dict[str, float] = {}
        for cell_value in _require_array(point_value, "raw archive point"):
            cell = _require_array(cell_value, "raw archive point cell")
            if len(cell) != 2 or any(type(value) is not str for value in cell):
                raise RuntimeError("raw archive point cell is malformed")
            point[cell[0]] = float.fromhex(cell[1])
        point_values.append(point)
    snapshot = FrozenArchiveWaveSnapshot2D.create(
        objectives=OBJECTIVES,
        reference_point=reference,
        archive_points=tuple(point_values),
    )
    raw_expected = receipt.get("raw_oriented_base_hypervolume_hex")
    if (
        type(raw_expected) is not str
        or not math.isclose(
            snapshot.base_hypervolume,
            float.fromhex(raw_expected),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("reconstructed raw hypervolume differs from source trace")
    normalized_expected = receipt.get("base_hypervolume_hex")
    if (
        type(normalized_expected) is not str
        or not math.isclose(
            snapshot.base_hypervolume / RAW_HYPERVOLUME_SCALE,
            float.fromhex(normalized_expected),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("reconstructed normalized hypervolume differs from trace")
    return {
        "reference_point": reference_record,
        "archive_points": raw_points,
        "snapshot_hash": snapshot.snapshot_hash,
    }


def _build_plan(
    *,
    events: list[dict[str, Any]],
    variation_topology: CampaignVariationTopology,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates = _candidate_index(events)
    candidate_by_configuration = {
        value["configuration_sha256"]: value for value in candidates.values()
    }
    archive_event_by_generation: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("kind") != "archive_utility_frozen":
            continue
        payload = _require_object(event.get("payload"), "archive utility payload")
        utility = _require_object(payload.get("archive_utility"), "archive utility")
        generation = utility.get("generation")
        if type(generation) is not int or generation in archive_event_by_generation:
            raise RuntimeError("archive utility generations must be exact and unique")
        archive_event_by_generation[generation] = event

    evaluation_plan: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    recorded_selected = 0
    recorded_prior_outcome = 0
    support_configurations: set[str] = set()
    for event in events:
        if event.get("kind") != "stage_sealed":
            continue
        payload = _require_object(event.get("payload"), "stage payload")
        stage = _require_object(payload.get("stage_receipt"), "stage receipt")
        generation = stage.get("generation")
        if type(generation) is not int or generation % 2 == 0:
            continue
        result = _require_object(stage.get("result"), "stage result")
        wave_receipts = {
            value["selection_call_id"]: value
            for value in (
                _require_object(row, "portfolio wave receipt")
                for row in _require_array(
                    result.get("portfolio_wave_receipts"),
                    "portfolio wave receipts",
                )
            )
        }
        audits = _require_array(stage.get("selector_audits"), "selector audits")
        if len(audits) != 2 or len(wave_receipts) != 2:
            raise RuntimeError("each odd generation must contain two exact lanes")
        snapshot_record = _archive_snapshot_record(
            archive_event_by_generation[generation]
        )
        for audit_ordinal, audit_value in enumerate(audits, start=1):
            audit = _require_object(audit_value, "selector audit")
            plaintext = _require_object(
                audit.get("plaintext_audit"), "plaintext selector audit"
            )
            call_id = plaintext.get("selector_call_id")
            if type(call_id) is not str or call_id not in wave_receipts:
                raise RuntimeError("selector audit has no exact portfolio receipt")
            wave_receipt = wave_receipts[call_id]
            frame = _prompt_frame(plaintext.get("request_text"))
            constraints = _require_object(
                frame.get("proposal_constraints"),
                "selector proposal constraints",
            )
            require_disjoint = constraints.get(
                "require_pairwise_disjoint_parent_patches"
            )
            if type(require_disjoint) is not bool:
                raise RuntimeError("selector disjoint constraint is malformed")
            min_distinct_families = constraints.get("min_distinct_families")
            if min_distinct_families is not None and type(
                min_distinct_families
            ) is not int:
                raise RuntimeError("selector family constraint is malformed")
            context = _require_object(frame.get("context"), "selector context")
            parent = _require_object(context.get("parent"), "selector parent")
            rendered = parent.get("rendered")
            if type(rendered) is not str:
                raise RuntimeError("selector parent omitted its rendered configuration")
            sequence = rendered.split(" -> ")
            if len(sequence) != 20:
                raise RuntimeError("selector parent does not contain 20 BOiLS actions")
            parent_configuration = {"sequence": sequence}
            contract = bind_finite_variation_catalog(
                variation_topology.decorate(BoilsFiniteVariationCatalog()),
                freeze_json(parent_configuration),
            )
            finite_contract = _require_object(
                frame.get("finite_variation_contract"),
                "finite variation contract",
            )
            if (
                finite_contract.get("parent_configuration_sha256")
                != contract.parent_configuration_sha256
            ):
                raise RuntimeError("prompt parent differs from reconstructed parent")
            parent_candidate_id = wave_receipt.get("parent_candidate_id")
            if (
                type(parent_candidate_id) is not str
                or parent_candidate_id not in candidates
            ):
                raise RuntimeError("wave receipt names an unknown parent")
            parent_objectives = _objective_values(candidates[parent_candidate_id])

            response_text = plaintext.get("response_text")
            if type(response_text) is not str:
                raise RuntimeError("selector audit omitted its trusted response")
            response = _require_object(
                json.loads(response_text), "trusted selector response"
            )
            supplemental = _require_object(
                response.get("supplemental_selector_audit"),
                "supplemental selector audit",
            )
            if (
                supplemental.get("decision_sha256") != audit.get("decision_sha256")
                or supplemental.get("request_sha256") != audit.get("request_sha256")
            ):
                raise RuntimeError("trusted response does not join its selector audit")
            supplemental_payload = _require_object(
                supplemental.get("payload"), "supplemental selector payload"
            )
            allocation = _require_object(
                supplemental_payload.get("allocation"), "K8-to-K4 allocation"
            )
            slate = _require_object(
                supplemental_payload.get("calibrated_slate"), "calibrated slate"
            )
            members = _require_array(slate.get("members"), "calibrated members")
            selected = _require_array(allocation.get("selected"), "allocated members")
            score_rows = _require_array(
                allocation.get("score_rows"), "allocation score rows"
            )
            if len(members) != 8 or len(score_rows) != 8 or len(selected) != 4:
                raise RuntimeError("source audit is not an exact K8-to-K4 decision")
            selected_ids = {
                _require_object(value, "allocated member").get("option_id")
                for value in selected
            }
            if any(type(value) is not str for value in selected_ids):
                raise RuntimeError("allocated member omitted an option ID")
            selected_by_option_id = {
                _require_object(value, "allocated member")["option_id"]: _require_object(
                    value, "allocated member"
                )
                for value in selected
            }
            score_by_option_id = {
                _require_object(value, "allocation score row")[
                    "option_id"
                ]: _require_object(value, "allocation score row")
                for value in score_rows
            }

            selected_outcomes: dict[str, dict[str, float]] = {}
            for attribution_value in _require_array(
                wave_receipt.get("action_attributions"), "action attributions"
            ):
                attribution = _require_object(
                    attribution_value, "action attribution"
                )
                selected_member = _require_object(
                    attribution.get("selected_member"), "selected member"
                )
                option_id = selected_member.get("option_id")
                candidate_id = attribution.get("candidate_id")
                if type(option_id) is not str or type(candidate_id) is not str:
                    raise RuntimeError("action attribution is malformed")
                selected_outcomes[option_id] = _objective_values(
                    candidates[candidate_id]
                )
            if set(selected_outcomes) != selected_ids:
                raise RuntimeError("selected outcomes differ from the K4 allocation")

            ordered: list[dict[str, Any]] = []
            for member_value in sorted(
                members,
                key=lambda value: _require_object(
                    value, "calibrated member"
                )["model_rank"],
            ):
                member = _require_object(member_value, "calibrated member")
                option_id = member.get("option_id")
                if type(option_id) is not str:
                    raise RuntimeError("calibrated member omitted option ID")
                option = contract.resolve(option_id)
                if (
                    option.identity_sha256 != member.get("option_identity_sha256")
                    or option.child_configuration_sha256
                    != member.get("phenotype_identity_sha256")
                ):
                    raise RuntimeError("current finite catalog differs from sealed K8")
                child_sha256 = option.child_configuration_sha256
                support_configurations.add(child_sha256)
                source_objectives = selected_outcomes.get(option_id)
                if source_objectives is not None:
                    recorded_selected += 1
                elif child_sha256 in candidate_by_configuration:
                    source_objectives = _objective_values(
                        candidate_by_configuration[child_sha256]
                    )
                    recorded_prior_outcome += 1
                configuration = thaw_json(option.child_configuration)
                row = {
                    "wave_ordinal": len(waves) + 1,
                    "cycle": generation,
                    "generation": generation,
                    "parent_slot": audit_ordinal - 1,
                    "request_sha256": audit["request_sha256"],
                    "parent_candidate_id": parent_candidate_id,
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "model_rank": member["model_rank"],
                    "option_id": option_id,
                    "option_identity_sha256": option.identity_sha256,
                    "configuration_sha256": child_sha256,
                    "configuration": configuration,
                    "predictions": member["predictions"],
                    "family": member["family"],
                    "locus_key": member["locus_key"],
                    "role_proposal": member["role_proposal"],
                    "structural_evidence": member["structural_evidence"],
                    "structural_posterior_scores": score_by_option_id[option_id],
                    "supporting_card_keys": member["supporting_card_keys"],
                    "calibrated_selected": option_id in selected_ids,
                    "selected_role": (
                        None
                        if option_id not in selected_by_option_id
                        else selected_by_option_id[option_id]["role"]
                    ),
                    "source_objectives": source_objectives,
                }
                row["evaluator_configuration_sha256"] = config_sha256(configuration)
                ordered.append(row)
                if source_objectives is None:
                    evaluation_plan.append(row)
            ordered_option_ids = tuple(row["option_id"] for row in ordered)
            feasible_k4_option_id_sets = [
                list(subset)
                for subset in combinations(ordered_option_ids, 4)
                if (
                    not require_disjoint
                    or finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
                        contract,
                        subset,
                        portfolio_size=4,
                        min_distinct_families=min_distinct_families,
                    )
                )
            ]
            if not feasible_k4_option_id_sets:
                raise RuntimeError("recorded K8 support has no feasible K4 subset")
            if frozenset(selected_ids) not in {
                frozenset(value) for value in feasible_k4_option_id_sets
            }:
                raise RuntimeError("recorded K4 allocation violates hard feasibility")
            waves.append(
                {
                    "wave_ordinal": len(waves) + 1,
                    "cycle": generation,
                    "generation": generation,
                    "request_sha256": audit["request_sha256"],
                    "parent_candidate_id": parent_candidate_id,
                    "parent_configuration": parent_configuration,
                    "parent_objectives": parent_objectives,
                    "archive_reward_snapshot": snapshot_record,
                    "feasible_k4_option_id_sets": feasible_k4_option_id_sets,
                    "members": ordered,
                }
            )
    if len(waves) != 6 or sum(len(value["members"]) for value in waves) != 48:
        raise RuntimeError("expected six K8 waves and 48 proposal occurrences")
    if recorded_selected != 24:
        raise RuntimeError("expected exactly 24 recorded selected outcomes")
    if len(evaluation_plan) + recorded_selected + recorded_prior_outcome != 48:
        raise RuntimeError("K8 outcome accounting is not exhaustive")
    unique_fresh_evaluations = len(
        {
            row["evaluator_configuration_sha256"]
            for row in evaluation_plan
        }
    )
    if len(support_configurations) < 24:
        raise RuntimeError("recorded K8 support contains implausibly few phenotypes")
    return evaluation_plan, waves, {
        "recorded_selected_outcomes": recorded_selected,
        "recorded_prior_outcomes": recorded_prior_outcome,
        "fresh_evaluation_occurrences": len(evaluation_plan),
        "fresh_evaluations": unique_fresh_evaluations,
        "completed_support": 48,
        "unique_support_configurations": len(support_configurations),
        "support_collision_occurrences": 48 - len(support_configurations),
    }


def _outcome_index(
    waves: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    result = {
        (value["request_sha256"], value["option_id"]): value["objectives"]
        for value in evaluations
    }
    for wave in waves:
        for member in wave["members"]:
            source = member["source_objectives"]
            if source is not None:
                result[(wave["request_sha256"], member["option_id"])] = source
    if len(result) != 48:
        raise RuntimeError("completed K8 outcome index must contain 48 rows")
    return result


def _unique_evaluation_plan(
    evaluation_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated deterministic phenotypes before physical execution.

    Proposal occurrence identity remains request/option scoped.  Physical
    evaluator identity is configuration scoped, so a phenotype proposed in
    two waves must consume one simulator call and later be expanded back to
    both authenticated occurrence identities.
    """

    result: dict[str, dict[str, Any]] = {}
    for row in evaluation_plan:
        key = row["evaluator_configuration_sha256"]
        prior = result.get(key)
        if prior is None:
            result[key] = row
            continue
        if (
            prior["configuration_sha256"] != row["configuration_sha256"]
            or prior["configuration"] != row["configuration"]
        ):
            raise RuntimeError(
                "one evaluator configuration identity resolves inconsistently"
            )
    return list(result.values())


def _expand_evaluation_occurrences(
    evaluation_plan: list[dict[str, Any]],
    unique_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join one physical result to every request-scoped proposal occurrence."""

    by_configuration: dict[str, dict[str, Any]] = {}
    for evaluation in unique_evaluations:
        key = evaluation["evaluator_configuration_sha256"]
        prior = by_configuration.get(key)
        if prior is not None and (
            prior["objectives"] != evaluation["objectives"]
            or prior["evaluation"] != evaluation["evaluation"]
        ):
            raise RuntimeError("duplicate physical evaluator outcomes disagree")
        by_configuration[key] = evaluation
    expected = {
        row["evaluator_configuration_sha256"] for row in evaluation_plan
    }
    if set(by_configuration) != expected:
        raise RuntimeError("physical evaluator outcomes do not cover the unique plan")

    result: list[dict[str, Any]] = []
    for row in evaluation_plan:
        evaluation = by_configuration[row["evaluator_configuration_sha256"]]
        result.append(
            {
                **evaluation,
                "wave_ordinal": row["wave_ordinal"],
                "request_sha256": row["request_sha256"],
                "model_rank": row["model_rank"],
                "option_id": row["option_id"],
                "option_identity_sha256": row["option_identity_sha256"],
                "configuration_sha256": row["configuration_sha256"],
            }
        )
    return result


def _mean_unique_evaluation_elapsed_s(
    evaluations: list[dict[str, Any]],
) -> float:
    by_configuration: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        key = row["evaluator_configuration_sha256"]
        prior = by_configuration.get(key)
        if prior is not None and prior["evaluation"] != row["evaluation"]:
            raise RuntimeError("expanded evaluator evidence differs by occurrence")
        by_configuration[key] = row
    return fmean(
        row["evaluation"]["elapsed_s"] for row in by_configuration.values()
    )


def _replay_evaluations(
    source_dir: Path,
    evaluation_plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Reuse one finalized diagnostic's evaluator rows with exact identity checks."""

    source_dir = source_dir.expanduser().resolve(strict=True)
    source_finalization = verify_finalized_run_directory(source_dir)
    result = _require_object(
        json.loads((source_dir / "result.json").read_text(encoding="utf-8")),
        "replay diagnostic result",
    )
    if (
        result.get("status") != "completed"
        or result.get("provider_calls") != 0
        or result.get("api_key_reads") != 0
    ):
        raise RuntimeError("replay source is not a completed provider-free diagnostic")
    payload = _require_object(
        json.loads((source_dir / "evaluations.json").read_text(encoding="utf-8")),
        "replay evaluations",
    )
    rows = _require_array(payload.get("rows"), "replay evaluation rows")
    replay_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for value in rows:
        row = _require_object(value, "replay evaluation row")
        key = (row.get("request_sha256"), row.get("option_id"))
        if any(type(item) is not str for item in key) or key in replay_by_key:
            raise RuntimeError("replay evaluation identity is malformed or repeated")
        replay_by_key[key] = row
    plan_by_key = {
        (row["request_sha256"], row["option_id"]): row for row in evaluation_plan
    }
    if set(replay_by_key) != set(plan_by_key):
        raise RuntimeError("replay evaluations differ from the exact current plan")
    identity_fields = (
        "option_identity_sha256",
        "configuration_sha256",
        "evaluator_configuration_sha256",
    )
    for key, row in replay_by_key.items():
        plan = plan_by_key[key]
        if any(row.get(field) != plan[field] for field in identity_fields):
            raise RuntimeError("replay evaluation differs from the exact finite option")
        objectives = row.get("objectives")
        evaluation = row.get("evaluation")
        if type(objectives) is not dict or type(evaluation) is not dict:
            raise RuntimeError("replay evaluation omitted outcome evidence")
    batch_wall_s = payload.get("batch_wall_s")
    if type(batch_wall_s) not in (int, float) or isinstance(batch_wall_s, bool):
        raise RuntimeError("replay evaluation batch wall time is malformed")
    # Reading the finalization is part of the authenticated provenance gate.
    if type(source_finalization.get("finalization_sha256")) is not str:
        raise RuntimeError("replay source finalization is malformed")
    return [replay_by_key[key] for key in plan_by_key], float(batch_wall_s)


def _forecast_analysis(
    waves: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    outcomes = _outcome_index(waves, evaluations)
    groups: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    rows: list[dict[str, Any]] = []
    for wave in waves:
        parent = wave["parent_objectives"]
        for member in wave["members"]:
            outcome = outcomes[(wave["request_sha256"], member["option_id"])]
            for prediction_value in member["predictions"]:
                prediction = _require_object(prediction_value, "forecast prediction")
                metric_id = prediction["metric_id"]
                asserted = prediction["asserted_direction"]
                delta = outcome[metric_id] - parent[metric_id]
                observed = (
                    "decrease" if delta < 0.0 else "increase" if delta > 0.0 else "unchanged"
                )
                known = asserted != "unknown"
                correct = known and asserted == observed
                row = {
                    "generation": wave["generation"],
                    "wave_ordinal": wave["wave_ordinal"],
                    "parent_candidate_id": wave["parent_candidate_id"],
                    "model_rank": member["model_rank"],
                    "option_id": member["option_id"],
                    "family": member["family"],
                    "calibrated_selected": member["calibrated_selected"],
                    "metric_id": metric_id,
                    "confidence": prediction["confidence"],
                    "asserted_direction": asserted,
                    "observed_direction": observed,
                    "delta": delta,
                    "known": known,
                    "correct": correct,
                }
                rows.append(row)
                dimensions = (
                    ("overall", "all"),
                    ("generation", str(wave["generation"])),
                    ("selection", "selected" if member["calibrated_selected"] else "rejected"),
                    ("model_rank", str(member["model_rank"])),
                    ("family", member["family"]),
                    ("confidence", prediction["confidence"]),
                    ("metric", metric_id),
                )
                for key in dimensions:
                    values = groups[key]
                    values[0] += 1
                    values[1] += int(known)
                    values[2] += int(correct)
    summaries = []
    for (dimension, value), (total, known, correct) in sorted(groups.items()):
        summaries.append(
            {
                "dimension": dimension,
                "value": value,
                "total": total,
                "known": known,
                "correct": correct,
                "exact_direction_accuracy": None if known == 0 else correct / known,
            }
        )
    return {
        "schema_version": 1,
        "rows": rows,
        "summaries": summaries,
    }


def _normalized_allocation_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    aggregate = _require_object(analysis.get("aggregate"), "allocation aggregate")
    keys = (
        "calibrated_gain_sum",
        "direct_model_top4_gain_sum",
        "uniform_expected_gain_sum",
        "oracle_gain_sum",
        "calibrated_minus_direct_gain_sum",
        "calibrated_minus_uniform_expected_gain_sum",
    )
    return {
        "raw_hypervolume_scale": RAW_HYPERVOLUME_SCALE,
        **{
            key.replace("gain", "normalized_gain", 1): aggregate[key]
            / RAW_HYPERVOLUME_SCALE
            for key in keys
        },
    }


async def _main_async(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_finalization = verify_finalized_run_directory(source_run)
    events = _campaign_events(source_run)
    variation_topology = _variation_topology(source_run)
    evaluation_plan, waves, accounting = _build_plan(
        events=events,
        variation_topology=variation_topology,
    )
    replay_source = None
    if args.replay_evaluations is not None:
        replay_dir = args.replay_evaluations.expanduser().resolve(strict=True)
        if WORKSPACE_ROOT not in replay_dir.parents:
            raise RuntimeError("replay diagnostic must live inside the workspace")
        replay_finalization = verify_finalized_run_directory(replay_dir)
        replay_source = {
            "path": replay_dir.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": replay_finalization["finalization_sha256"],
            "recursive_content_sha256": replay_finalization[
                "recursive_content_sha256"
            ],
            "evaluations_sha256": _sha256_file(replay_dir / "evaluations.json"),
        }
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "running",
        "diagnostic": "complete_sealed_generic_campaign_k8_support",
        "source_run": {
            "path": source_run.relative_to(WORKSPACE_ROOT).as_posix(),
            "finalization_sha256": source_finalization["finalization_sha256"],
            "recursive_content_sha256": source_finalization[
                "recursive_content_sha256"
            ],
            "summary_sha256": _sha256_file(source_run / "summary.json"),
            "mutated": False,
        },
        "workload": {
            "id": "boils-abc-pinned-panel",
            "panel": ["log2"],
            "variation_topology": variation_topology.to_record(),
            "accounting": accounting,
            "affinity_sets": [list(value) for value in args.affinity_sets],
        },
        "replay_evaluation_source": replay_source,
        "claim_boundary": {
            "provider_calls": 0,
            "api_key_reads": 0,
            "campaign_counterfactual": False,
            "fixed_parent_fixed_k8_local_mechanism_diagnostic": True,
            "paper_ready_efficacy": False,
            "evaluator_rows_replayed": args.replay_evaluations is not None,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "source_identity": source_identity(
            (
                Path(__file__),
                AGENT_EVOLVE_ROOT
                / "examples/development/evaluate_boils_calibrated_slate_counterfactual.py",
                AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
                AGENT_EVOLVE_ROOT
                / "examples/benchmarks/boils_abc/finite_variation_catalog.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/reward/frozen_wave_archive.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/campaign_variation_topology.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/variation/compositional_finite_catalog.py",
                AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
            ),
            relative_to=WORKSPACE_ROOT,
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_json_atomic(
        output_dir / "evaluation_plan.json",
        {"schema_version": 1, "accounting": accounting, "rows": evaluation_plan},
    )
    if args.replay_evaluations is None:
        unique_evaluation_plan = _unique_evaluation_plan(evaluation_plan)
        unique_evaluations, wall_s = await _evaluate_rejected(
            unique_evaluation_plan,
            affinity_sets=args.affinity_sets,
        )
        evaluations = _expand_evaluation_occurrences(
            evaluation_plan,
            unique_evaluations,
        )
        unique_evaluation_count = len(unique_evaluations)
        evaluation_execution_mode = "fresh_real_evaluator_execution"
    else:
        evaluations, wall_s = _replay_evaluations(
            args.replay_evaluations,
            evaluation_plan,
        )
        unique_evaluation_count = len(
            {
                row["evaluator_configuration_sha256"]
                for row in evaluations
            }
        )
        evaluation_execution_mode = "authenticated_finalized_evaluator_replay"
    allocation = _allocation_analysis(waves, evaluations)
    normalized = _normalized_allocation_summary(allocation)
    forecasts = _forecast_analysis(waves, evaluations)
    write_json_atomic(
        output_dir / "evaluations.json",
        {
            "schema_version": 1,
            "planned_missing_source_outcome_occurrence_count": len(evaluation_plan),
            "planned_missing_source_outcome_count": unique_evaluation_count,
            "outcome_occurrence_count": len(evaluations),
            "fresh_evaluation_count": (
                unique_evaluation_count if args.replay_evaluations is None else 0
            ),
            "replayed_evaluation_count": (
                0 if args.replay_evaluations is None else unique_evaluation_count
            ),
            "execution_mode": evaluation_execution_mode,
            "batch_wall_s": wall_s,
            "mean_evaluation_elapsed_s": _mean_unique_evaluation_elapsed_s(
                evaluations
            ),
            "rows": evaluations,
        },
    )
    write_json_atomic(output_dir / "allocation_analysis.json", allocation)
    write_json_atomic(output_dir / "forecast_analysis.json", forecasts)
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "source_run_mutated": False,
        "provider_calls": 0,
        "api_key_reads": 0,
        "accounting": accounting,
        "evaluation_execution_mode": evaluation_execution_mode,
        "diagnostic_fresh_evaluations_executed": (
            unique_evaluation_count if args.replay_evaluations is None else 0
        ),
        "diagnostic_evaluations_replayed": (
            0 if args.replay_evaluations is None else unique_evaluation_count
        ),
        "diagnostic_outcome_occurrences": len(evaluations),
        "batch_wall_s": wall_s,
        "mean_evaluation_elapsed_s": _mean_unique_evaluation_elapsed_s(
            evaluations
        ),
        "allocation_aggregate": allocation["aggregate"],
        "normalized_allocation_aggregate": normalized,
        "forecast_overall": next(
            value
            for value in forecasts["summaries"]
            if value["dimension"] == "overall" and value["value"] == "all"
        ),
        "claim_scope": allocation["claim_scope"],
    }
    write_json_atomic(output_dir / "result.json", result)
    finalization = finalize_run_directory(output_dir, status="completed")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": finalization["finalization_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replay-evaluations",
        type=Path,
        help="finalized prior K8 diagnostic whose exact evaluator rows are reused",
    )
    parser.add_argument(
        "--affinity-sets",
        type=_parse_affinity_sets,
        default=_parse_affinity_sets(",".join(str(value) for value in range(16))),
        help="comma-separated CPU IDs; each becomes one exclusive evaluator slot",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main_async(args))
    except BaseException as exc:
        output_dir = args.output_dir.expanduser().resolve(strict=False)
        manifest = output_dir / "manifest.json"
        finalized = output_dir / "finalized.json"
        if manifest.is_file() and not finalized.exists():
            write_json_atomic(
                output_dir / "failed.json",
                {
                    "schema_version": 1,
                    "status": "failed_harness",
                    "failure_type": type(exc).__name__,
                    "safe_message": "generic K8 diagnostic failed before a scientific result",
                    "provider_calls": 0,
                    "api_key_reads": 0,
                    "source_run_mutated": False,
                },
            )
            finalize_run_directory(output_dir, status="failed_harness")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
