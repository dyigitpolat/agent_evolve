"""Outcome-blind fresh-parent transfer blocks for the Stage-B selector.

The generic matched finite-action planner remains unchanged.  This benchmark
policy ranks every safe, historically unseen Airfoil parent by a public hash,
excludes the two G3 source parents, and binds one schedule position to the
existing learned card and K=8 same-support A/U experiment.  Parent ranking
never reads CFD outcomes, provider evidence, or credentials.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from agent_evolve.agentic import (
    CandidateId,
    EngineFiniteActionRequest,
    OperatorKind,
    TaskKeyedUniformFiniteActionPolicy,
    thaw_json,
)
from examples.benchmarks.engibench_airfoil.problem_def import normalize_candidate
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    CandidateMaterialization,
    PARENT_GRID_NONCES,
    SelectedParent,
    parent_grid_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AirfoilG3RuntimeInputs,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
    AIRFOIL_V8_STAGE_B_CATALOG_ID,
    AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    AirfoilV8StageBError,
    AirfoilV8StageBInputs,
    airfoil_v8_stage_b_readiness_record,
    compose_airfoil_v8_stage_b_inputs,
)


PANEL_POLICY_ID = "airfoil_v9_stage_b_transfer_parent_panel"
PANEL_POLICY_VERSION = 1
PANEL_RANK_DOMAIN = b"agent-evolve:airfoil-v9-stage-b-transfer-parent-panel:v1\x00"
_PANEL_DEFINITION_DOMAIN = (
    b"agent-evolve:airfoil-v9-stage-b-transfer-parent-panel:def:v1\x00"
)
_BLOCK_DEFINITION_DOMAIN = b"agent-evolve:airfoil-v9-stage-b-transfer:def:v1\x00"
_TASK_DOMAIN = b"agent-evolve:airfoil-v9-stage-b-transfer-task:v1\x00"
_SCHEDULE_DOMAIN = b"agent-evolve:airfoil-v9-stage-b-transfer-schedule:v1\x00"
_COMMIT_DOMAIN = b"agent-evolve:airfoil-v9-stage-b-transfer-commit:v1\x00"

_PANEL_POLICY_DEFINITION = {
    "policy_id": PANEL_POLICY_ID,
    "policy_version": PANEL_POLICY_VERSION,
    "candidate_source": "airfoil_v7_frozen_parent_grid_nonces_1_through_255",
    "admission": [
        "candidate_materialization_and_no_cfd_geometry_pass",
        "near_neutral_coefficient_and_alpha_bounds_pass",
        "historical_configuration_candidate_and_phenotype_denylist_pass",
        "not_one_of_the_two_g3_source_parents",
    ],
    "rank": "sha256(domain,canonical_public_basis_and_candidate_identity)",
    "order": "rank_digest_then_nonce",
    "outcomes_read": False,
    "credentials_read": False,
}
PANEL_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _PANEL_DEFINITION_DOMAIN
    + json.dumps(
        _PANEL_POLICY_DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


def _panel_basis(source: AirfoilG3RuntimeInputs) -> dict[str, object]:
    source.preparation.__post_init__()
    excluded = sorted(
        (
            source.preparation.diagnostic_parent.candidate.configuration_sha256,
            source.preparation.heldout_parent.candidate.configuration_sha256,
        )
    )
    if len(set(excluded)) != 2:
        raise AirfoilV8StageBError("G3 source parents collide")
    return {
        "policy_id": PANEL_POLICY_ID,
        "policy_version": PANEL_POLICY_VERSION,
        "source_run_id": "airfoil_v7_g3_20260715t1204z",
        "learned_card_content_sha256": AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
        "historical_membership_sha256": (
            source.preparation.membership.membership_sha256
        ),
        "excluded_g3_parent_configuration_sha256s": excluded,
        "current_outcome_access": False,
    }


def rank_airfoil_v9_transfer_parent_panel(
    source: AirfoilG3RuntimeInputs,
) -> tuple[SelectedParent, ...]:
    """Return the complete public ranking of safe non-source parents."""

    if type(source) is not AirfoilG3RuntimeInputs:
        raise TypeError("source must be exact AirfoilG3RuntimeInputs")
    # The v8 card replay deliberately adds one exact revision to the runtime
    # memory.  Parent selection depends only on the immutable release
    # preparation, so validate that object rather than rejecting the later
    # readiness projection for its expected memory addition.
    source.preparation.__post_init__()
    basis = _panel_basis(source)
    excluded = set(basis["excluded_g3_parent_configuration_sha256s"])
    membership = source.preparation.membership
    eligible: list[tuple[str, int, CandidateMaterialization]] = []
    for nonce in PARENT_GRID_NONCES:
        try:
            materialized = CandidateMaterialization.from_configuration(
                f"transfer_parent_grid_nonce_{nonce:03d}",
                parent_grid_candidate(nonce),
            )
        except (TypeError, ValueError):
            continue
        candidate = normalize_candidate(thaw_json(materialized.configuration))
        if any(abs(value) > 0.022 for value in candidate["upper_coefficients"]):
            continue
        if any(abs(value) > 0.022 for value in candidate["lower_coefficients"]):
            continue
        if any(not 0.5 <= value <= 9.5 for value in candidate["alpha_deg"]):
            continue
        if membership.rejects(
            configuration_sha256=materialized.configuration_sha256,
            candidate_sha256_value=materialized.candidate_sha256,
            phenotype_value_sha256=materialized.phenotype_value_sha256,
        ):
            continue
        if materialized.configuration_sha256 in excluded:
            continue
        candidate_record = {
            **basis,
            "nonce": nonce,
            "configuration_sha256": materialized.configuration_sha256,
            "candidate_sha256": materialized.candidate_sha256,
            "phenotype_value_sha256": materialized.phenotype_value_sha256,
        }
        rank = _hash(PANEL_RANK_DOMAIN, candidate_record)
        eligible.append((rank, nonce, materialized))
    eligible.sort(key=lambda value: (value[0], value[1]))
    if not eligible:
        raise AirfoilV8StageBError("fresh transfer panel is empty")
    panel = tuple(
        SelectedParent(
            role=f"T_{index:03d}",
            nonce=nonce,
            selection_sha256=rank,
            candidate=materialized,
        )
        for index, (rank, nonce, materialized) in enumerate(eligible)
    )
    if len({row.candidate.configuration_sha256 for row in panel}) != len(panel):
        raise AirfoilV8StageBError("transfer panel configurations collide")
    if len({row.candidate.phenotype_value_sha256 for row in panel}) != len(panel):
        raise AirfoilV8StageBError("transfer panel phenotypes collide")
    return panel


def _block_bindings(
    source: AirfoilG3RuntimeInputs,
    *,
    panel_index: int,
) -> tuple[SelectedParent, str, str, str, str]:
    if type(panel_index) is not int or panel_index < 0:
        raise ValueError("panel_index must be a non-negative exact integer")
    panel = rank_airfoil_v9_transfer_parent_panel(source)
    try:
        parent = panel[panel_index]
    except IndexError as exc:
        raise ValueError("panel_index lies outside the fresh parent panel") from exc
    definition = _hash(
        _BLOCK_DEFINITION_DOMAIN,
        {
            "schema_version": 1,
            "panel_policy_definition_sha256": PANEL_POLICY_DEFINITION_SHA256,
            "panel_basis": _panel_basis(source),
            "panel_cardinality": len(panel),
            "panel_index": panel_index,
            "parent": parent.to_record(),
            "card_content_sha256": AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
            "support_cardinality": AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
            "arms": ["A:model_card_choice", "U:prospective_uniform_same_support"],
            "current_outcome_access": False,
        },
    )
    task = _hash(
        _TASK_DOMAIN,
        {
            "definition_sha256": definition,
            "parent_configuration_sha256": parent.candidate.configuration_sha256,
            "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
        },
    )
    schedule = _hash(
        _SCHEDULE_DOMAIN,
        {
            "definition_sha256": definition,
            "panel_index": panel_index,
            "parent_selection_sha256": parent.selection_sha256,
        },
    )
    commit = _hash(
        _COMMIT_DOMAIN,
        {
            "definition_sha256": definition,
            "task_sha256": task,
            "schedule_seed_sha256": schedule,
            "new_transfer_outcomes_observed": False,
        },
    )
    return parent, definition, task, schedule, commit


def compose_airfoil_v9_stage_b_transfer_inputs(
    source: AirfoilG3RuntimeInputs,
    *,
    panel_index: int,
) -> AirfoilV8StageBInputs:
    """Bind the unchanged generic Stage-B loop to one fresh panel parent."""

    parent, _definition, task, schedule, commit = _block_bindings(
        source,
        panel_index=panel_index,
    )
    base = compose_airfoil_v8_stage_b_inputs(source)
    seed = thaw_json(parent.candidate.configuration)
    if type(seed) is not dict:
        raise AirfoilV8StageBError("fresh transfer parent is not an object")
    factory = replace(
        base.planner_factory,
        task_sha256=task,
        pre_outcome_phase_commit_sha256=commit,
        uniform_policy=TaskKeyedUniformFiniteActionPolicy(
            schedule_seed_sha256=schedule,
        ),
        phase=f"airfoil_v9_stage_b_transfer_{panel_index:03d}",
    )
    inputs = replace(
        base,
        seed_configuration=seed,
        planner_factory=factory,
    )
    inputs.__post_init__()
    return inputs


def airfoil_v9_stage_b_transfer_readiness_record(
    source: AirfoilG3RuntimeInputs,
    inputs: AirfoilV8StageBInputs,
    *,
    panel_index: int,
) -> dict[str, object]:
    """Expose parent freshness and the sealed U choice before live outcomes."""

    parent, definition, task, schedule, commit = _block_bindings(
        source,
        panel_index=panel_index,
    )
    inputs.__post_init__()
    if inputs.seed_configuration != thaw_json(parent.candidate.configuration):
        raise AirfoilV8StageBError("transfer inputs use another panel parent")
    if (
        inputs.planner_factory.task_sha256 != task
        or inputs.planner_factory.pre_outcome_phase_commit_sha256 != commit
        or inputs.planner_factory.uniform_policy.schedule_seed_sha256 != schedule
    ):
        raise AirfoilV8StageBError("transfer planner bindings drifted")
    compiled = inputs.benchmark.compile_registered_hypothesis_treatment(
        catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
        parent_candidate_id=CandidateId(
            "candidate_airfoil_g3_runtime_000001"
        ),
        parent_configuration=inputs.seed_configuration,
        entry=inputs.learned_card,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256=(
            inputs.planner_factory.context_projection_sha256
        ),
        endpoint_definition_sha256=(
            inputs.planner_factory.endpoint_definition_sha256
        ),
    )
    authority, _ = inputs.benchmark.compile_finite_action_set(
        compiled_anchor=compiled,
        required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        source_mode=inputs.planner_factory.source_mode,
    )
    rank = inputs.planner_factory.uniform_policy.freeze_rank(
        authority,
        task_sha256=task,
        pre_outcome_phase_commit_sha256=commit,
    )
    decision = inputs.planner_factory.uniform_policy.choose(
        EngineFiniteActionRequest(authority=authority, prospective_rank=rank)
    )
    record = airfoil_v8_stage_b_readiness_record(inputs)
    record.update(
        {
            "schema_version": 2,
            "claim_boundary": (
                "fresh_parent_development_block_not_yet_replicated_paper_evidence"
            ),
            "definition_sha256": definition,
            "transfer_parent": {
                "panel_policy_id": PANEL_POLICY_ID,
                "panel_policy_version": PANEL_POLICY_VERSION,
                "panel_policy_definition_sha256": (
                    PANEL_POLICY_DEFINITION_SHA256
                ),
                "panel_cardinality": len(rank_airfoil_v9_transfer_parent_panel(source)),
                "panel_index": panel_index,
                "selection_sha256": parent.selection_sha256,
                "nonce": parent.nonce,
                "configuration_sha256": parent.candidate.configuration_sha256,
                "candidate_sha256": parent.candidate.candidate_sha256,
                "phenotype_value_sha256": parent.candidate.phenotype_value_sha256,
                "historical_membership_rejected": False,
                "g3_source_parent": False,
                "outcomes_read_by_selection": False,
            },
            "prospective_uniform": {
                "schedule_seed_sha256": schedule,
                "task_sha256": task,
                "pre_outcome_phase_commit_sha256": commit,
                "rank_token_sha256": rank.token_sha256,
                "selected_ordinal": decision.selected_ordinal,
                "option_id": decision.option_id,
                "propensity": [1, AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY],
                "outcomes_read": False,
            },
        }
    )
    return record


__all__ = [
    "PANEL_POLICY_DEFINITION_SHA256",
    "PANEL_POLICY_ID",
    "PANEL_POLICY_VERSION",
    "airfoil_v9_stage_b_transfer_readiness_record",
    "compose_airfoil_v9_stage_b_transfer_inputs",
    "rank_airfoil_v9_transfer_parent_panel",
]
