"""Provider/CFD-free proof for Airfoil v10's frozen input authority matrix."""

from __future__ import annotations

import ast
import re
import shutil
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import pytest

from agent_evolve.application.insight_memory import context_stratum_hash
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    compose_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import AirfoilV7Problem
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    AIRFOIL_V10_AUTHORITY_MATRIX_ORDER,
    AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS,
    AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
    AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256,
    AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256,
    AIRFOIL_V10_MULTI_OPTION_PHASE,
    AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256,
    AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256,
    AIRFOIL_V10_MULTI_OPTION_TASK_SHA256,
    AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256,
    AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256,
    AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON,
    AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE,
    AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256,
    AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT,
    AIRFOIL_V10_PROMOTION_REASON,
    AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE,
    AirfoilV10MultiOptionInputError,
    airfoil_v10_gpt_profile_config_record,
    airfoil_v10_multi_option_readiness_record,
    authenticate_airfoil_v10_promotion_evidence,
    authenticate_airfoil_v10_other_card_promotion_evidence,
    build_airfoil_v10_gpt_openrouter_config,
    compose_airfoil_v10_multi_option_inputs,
)


class _NoRawCFD:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_raw(self, configuration):
        del configuration
        self.calls += 1
        raise AssertionError("v10 input preflight must not invoke CFD")


@cache
def _prepared() -> release.AirfoilG3ReleasePreparation:
    return release.prepare_release()


def _forbidden_reasoning_key_paths(
    value: object,
    *,
    path: str = "$",
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert type(key) is str
            if key.casefold() in {"mode", "pro"}:
                found.append(f"{path}.{key}")
            found.extend(
                _forbidden_reasoning_key_paths(child, path=f"{path}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(
                _forbidden_reasoning_key_paths(
                    child,
                    path=f"{path}[{index}]",
                )
            )
    return tuple(found)


def test_airfoil_v10_seals_two_cards_by_two_seeds_without_provider_or_cfd() -> (
    None
):
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    raw = _NoRawCFD()
    source = compose_airfoil_g3_runtime_inputs(
        problem=AirfoilV7Problem(raw_problem=raw),
        preparation=preparation,
        diagnostic_permutation=permutation,
    )
    source_entries_before = source.memory.entries
    source_runtime_sha256 = source.runtime_inputs_sha256

    inputs = compose_airfoil_v10_multi_option_inputs(source)
    readiness = airfoil_v10_multi_option_readiness_record(inputs)

    # Replay occurs in an independent v10 bank. The frozen G3 input remains a
    # valid value object and can still be checked after v2 has been installed.
    source.__post_init__()
    assert source.memory.entries == source_entries_before
    assert len(source.memory.entries) == 3
    assert len(inputs.memory.entries) == 5
    assert inputs.memory is not source.memory
    assert inputs.id_factory is not source.id_factory
    assert inputs.source_runtime_inputs_sha256 == source_runtime_sha256

    learned, other = inputs.active_cards
    assert (
        learned.reference.insight_id.value,
        learned.reference.version,
        learned.draft.content_sha256,
    ) == (
        "insight_airfoil_twostage_cards_000002",
        2,
        "ecc66ffd283a70b675551e2ca8dcbd0dbd803135ccace8d4c06101e26494972d",
    )
    assert (
        other.reference.insight_id.value,
        other.reference.version,
        other.draft.content_sha256,
    ) == (
        "insight_airfoil_twostage_cards_000007",
        2,
        "425e23a51a5ed9831defe58faf22095f302ecd06aa9d23138c4f7a8f7271cfb5",
    )
    assert learned.evidence_lineage is not None
    assert learned.evidence_lineage.identity_sha256 == (
        "7353690f460f208a4a27a318551fda4ae309526af3390b9c4fc06c651e6c9bdc"
    )
    assert learned.lifecycle_state.value == "promoted"
    assert learned.retrievable is True
    assert other.lifecycle_state.value == "promoted"
    assert other.retrievable is True
    assert other.origin.value == "manual"
    assert other.applicable_operator_kinds == ("typed_mutation",)
    assert set(other.draft.affected_paths) == {
        "$.alpha_deg[0]",
        "$.alpha_deg[1]",
        "$.alpha_deg[2]",
    }
    assert other.evidence_lineage is None
    assert len(other.relations) == 1
    assert other.relations[0].kind.value == "revises"
    assert other.relations[0].target.version == 1
    source_other = next(
        entry
        for entry in inputs.memory.entries
        if entry.reference.insight_id == other.reference.insight_id
        and entry.reference.version == 1
    )
    assert source_other.lifecycle_state.value == "quarantined"
    assert source_other.retrievable is False
    assert source_other.applicable_operator_kinds == ("mutation",)
    assert source_other.draft == other.draft
    assert len(inputs.memory.transitions) == 2
    transition = inputs.memory.transitions[0]
    assert transition.sequence == 1
    assert transition.reference == learned.reference
    assert transition.prior_state.value == "quarantined"
    assert transition.new_state.value == "promoted"
    assert transition.reason == AIRFOIL_V10_PROMOTION_REASON
    assert (
        transition.supporting_evidence
        == AIRFOIL_V10_PROMOTION_SUPPORTING_EVIDENCE
    )
    assert tuple(
        record.result_file_sha256 for record in inputs.promotion_evidence
    ) == (
        "aeb8745f607e4c1bdcb675c1dd4c318c2ab4767510597a7d3dd08e4b11cc0ad2",
        "58807f0943f7f54586f3c4050916ac0c4e71e047a841b6b3d827423955959d06",
        "fe8fa954eba694355d0c5b1dae6eb1ed8ee09a756efbb9e163f6635df0f7b333",
    )
    assert tuple(
        record.optimizer_result_sha256 for record in inputs.promotion_evidence
    ) == (
        "3282d5f5141f8a3780e8be1a65df9d0c4c57f3c165cc46f8064fb78e186bbf30",
        "350e5bfc3fbd181da6c2c28c7ca2e3c14c98d8acf96b8cccca4538215065a043",
        "c530aaa32d5a35b56e73e684050d15c1838281b919194e9c6c5cd4e47d796360",
    )
    assert all(
        float.fromhex(record.adaptive_minus_uniform_reward_hex) > 0.0
        for record in inputs.promotion_evidence
    )
    assert sum(
        record.fresh_transfer_parent for record in inputs.promotion_evidence
    ) == 2
    other_transition = inputs.memory.transitions[1]
    assert other_transition.sequence == 2
    assert other_transition.reference == other.reference
    assert other_transition.prior_state.value == "quarantined"
    assert other_transition.new_state.value == "promoted"
    assert other_transition.reason == AIRFOIL_V10_OTHER_CARD_PROMOTION_REASON
    assert other_transition.supporting_evidence == (
        AIRFOIL_V10_OTHER_CARD_PROMOTION_SUPPORTING_EVIDENCE
    )
    assert AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256 == (
        "d398d8e4fa1be53c61abdda0e14fdd6a9e0380b611e83426afe514befa8a0acc"
    )
    assert AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256 == (
        "5e7e121a2612683003a92828c379cd41e94d95cb77906ce20209c0ce79d9d92f"
    )
    assert AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256 == (
        "cfe728e031d6649e28cd1f1ba0027c3a796d89a043021a06466402175651510d"
    )
    assert inputs.promotion_evidence_set_sha256 == (
        "421423268521d24da1cf8e235c7386cea71bac4330b0f02b7f0cc33917f0d73d"
    )

    assert tuple(seed.role for seed in inputs.g0_seeds) == (
        "diagnostic_parent",
        "hypothesis_parent",
    )
    assert tuple(seed.configuration_sha256 for seed in inputs.g0_seeds) == (
        "4e17a2c2d5efce96e554858f4baad762de76626aec3df0d90ee63711545f9122",
        "cb601c3588ca6f17e527b8f7961c1b22e3ae12a138fd33c34bc869d1c6b852a0",
    )
    assert inputs.seed_configurations == source.seed_configurations

    # This is the context the generic planner independently derives in its
    # constructor. It is deliberately not the inherited G3 context.
    assert inputs.phase == AIRFOIL_V10_MULTI_OPTION_PHASE
    assert inputs.context_projection_sha256 == context_stratum_hash(
        problem_id=release.AIRFOIL_G3_RUNTIME_PROBLEM_ID,
        operator_kind="typed_mutation",
        phase=AIRFOIL_V10_MULTI_OPTION_PHASE,
    )
    assert inputs.context_projection_sha256 == (
        AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
    )
    old_g3_contexts = {
        request.context_projection_sha256
        for matrix in source.prepared_hypothesis_matrices
        for request in matrix.requests
    }
    assert old_g3_contexts == {release.CONTEXT_PROJECTION_SHA256}
    assert inputs.context_projection_sha256 not in old_g3_contexts

    assert tuple(
        (binding.seed_role, binding.card_role)
        for binding in inputs.authority_bindings
    ) == AIRFOIL_V10_AUTHORITY_MATRIX_ORDER
    assert tuple(
        binding.probe_candidate_id for binding in inputs.authority_bindings
    ) == AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS
    assert tuple(
        value.value for value in AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS
    ) == (
        "candidate_d0be50bf270b24e45046150d5aa8a9927484b8221d34f15cfe4863c093ad14e2",
        "candidate_83393b335a22651a7a2807d783b482eba6c14a71ef7087bfe202194001aea99a",
        "candidate_e2359cc5133bf261f8d7ba530ed3aefcec79931fad15b56f7c7711ac77c91730",
        "candidate_27d27cf546cbacff50c1262b51cf6d41399ea214e538b8898ddf6c229c33bbc0",
    )
    for probe in AIRFOIL_V10_AUTHORITY_PROBE_CANDIDATE_IDS:
        assert re.fullmatch(r"candidate_[0-9a-f]{64}", probe.value)
        assert not any(
            token in probe.value
            for token in (
                "diagnostic",
                "hypothesis",
                "learned",
                "other",
                "000002",
                "000007",
            )
        )

    all_children: set[str] = set()
    all_phenotypes: set[str] = set()
    for binding in inputs.authority_bindings:
        authority = binding.authority
        assert authority.current_outcome_access is False
        assert authority.support.cardinality == 8
        assert authority.support.parent_candidate_id == binding.probe_candidate_id
        assert authority.support.context_projection_sha256 == (
            AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
        )
        assert binding.compiled_treatment.request.context_projection_sha256 == (
            AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
        )
        assert binding.compiled_treatment.receipt.applicable
        assert len(binding.compiled_treatment.requirement.allowed_actions) == 1
        option_ids = {
            row.option.option_id for row in authority.support.options
        }
        child_hashes = {
            row.option.child_configuration_sha256
            for row in authority.support.options
        }
        phenotypes = {
            row.phenotype_identity_sha256 for row in authority.support.options
        }
        assert len(option_ids) == len(child_hashes) == len(phenotypes) == 8
        all_children.update(child_hashes)
        all_phenotypes.update(phenotypes)
    assert len(all_children) == len(all_phenotypes) == 32
    assert len(
        {binding.authority.support.support_sha256 for binding in inputs.authority_bindings}
    ) == 4
    assert len(
        {binding.authority.authority_sha256 for binding in inputs.authority_bindings}
    ) == 4

    for seed_role in ("diagnostic_parent", "hypothesis_parent"):
        learned_binding = inputs.authority_for(
            seed_role=seed_role,
            card_role="learned_v2",
        )
        other_binding = inputs.authority_for(
            seed_role=seed_role,
            card_role="other_migrated_v2",
        )
        learned_ids = {
            row.option.option_id
            for row in learned_binding.authority.support.options
        }
        other_ids = {
            row.option.option_id
            for row in other_binding.authority.support.options
        }
        assert learned_ids.isdisjoint(other_ids)
        assert learned_binding.authority.support.anchor_option_id == (
            "trim.p025.n025.p050"
        )
        assert other_binding.authority.support.anchor_option_id == (
            "trim.p050.n050.n050"
        )

    assert inputs.mate_choice == source.mate_choice
    assert inputs.mate_choice.option_id == "shape.camber_aft.n0030"
    assert inputs.mate_choice.selection_policy_id == "airfoil_v7_g3_sealed_mate"
    assert inputs.mate_configuration_dict != inputs.seed_configurations[1]
    assert readiness["orthogonal_mate"]["family"] == "shape_only"

    assert AIRFOIL_V10_MULTI_OPTION_DEFINITION_SHA256 == (
        "9e7fa2ec6559c10b227ee013b92974115b8dcf3af2269b51e4a4fd84e761b9d9"
    )
    assert AIRFOIL_V10_MULTI_OPTION_TASK_SHA256 == (
        "3a784bd5c264274a6de47d41d48d387ab7bfb0308a69d192fde4263dfccb1347"
    )
    assert AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256 == (
        "24057d3c74c3c1fb626c8a5c1ac7d5759086a2fc5bd79768a30a9eb4627a85e4"
    )
    assert AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256 == (
        "29dc325dfa8f5c4fb5332e9e5a2845219fc03da9900da846ac27e9abf3de4a71"
    )
    assert inputs.inputs_sha256 == (
        "82417a2082372162f06307da25d8a31032e2eca021fd16dc08ca77b9cd4336f4"
    )
    assert readiness["task_sha256"] == AIRFOIL_V10_MULTI_OPTION_TASK_SHA256
    assert readiness["schedule_sha256"] == (
        AIRFOIL_V10_MULTI_OPTION_SCHEDULE_SHA256
    )
    assert readiness["pre_outcome_commit_sha256"] == (
        AIRFOIL_V10_MULTI_OPTION_PRE_OUTCOME_COMMIT_SHA256
    )
    assert readiness["current_outcome_access"] is False
    promotion = readiness["learned_v2_promotion"]
    assert promotion["evidence_definition_sha256"] == (
        AIRFOIL_V10_PROMOTION_EVIDENCE_DEFINITION_SHA256
    )
    assert promotion["evidence_set_sha256"] == (
        inputs.promotion_evidence_set_sha256
    )
    assert promotion["all_results_adaptive_beats_uniform"] is True
    assert promotion["fresh_transfer_result_count"] == 2
    assert promotion["development_evidence_only"] is True
    assert promotion["paper_evidence_eligible"] is False
    assert promotion["transition"]["new_state"] == "promoted"
    assert promotion["transition"]["retrievable_after_transition"] is True
    other_promotion = readiness["other_migrated_v2_promotion"]
    assert other_promotion["evidence_definition_sha256"] == (
        AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_DEFINITION_SHA256
    )
    assert other_promotion["evidence_record_sha256"] == (
        AIRFOIL_V10_OTHER_CARD_PROMOTION_EVIDENCE_RECORD_SHA256
    )
    assert other_promotion["transition"]["new_state"] == "promoted"
    assert other_promotion["transition"]["retrievable_after_transition"] is True
    assert other_promotion["claim"] == "tested_retrievable_only"
    assert other_promotion["efficacy_claim"] is False
    assert other_promotion["paper_evidence_eligible"] is False
    assert readiness["claim_boundary"] == {
        "provider_called": False,
        "credentials_read": False,
        "physical_evaluator_called": False,
        "scientific_result_eligible": False,
        "meaning": "input and authority preflight only",
    }
    assert readiness["distinctness"] == {
        "authority_count": 4,
        "support_cardinality_each": 8,
        "globally_distinct_child_configurations": 32,
        "globally_distinct_phenotypes": 32,
        "within_seed_card_supports_disjoint": True,
    }
    assert raw.calls == 0


def test_airfoil_v10_promotion_evidence_is_file_authenticated_and_fail_closed(
    tmp_path: Path,
) -> None:
    records = authenticate_airfoil_v10_promotion_evidence()
    assert tuple(record.record_sha256 for record in records) == (
        "242c9c6c27cf10d0a7b2ac33ec891e2ed6edeceb05e31b43918ee34ce5a01af1",
        "2173bae3ad9a894ed06b63a3c38cca898bf5ac8e8cd282781ca9cf496434561b",
        "1ae2722d1a541bdcbb5f6ec6b37565e22b265ac71c5a3dcc88416697e0dea6e7",
    )

    for record in records:
        destination = tmp_path / record.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            AIRFOIL_V10_PROMOTION_EVIDENCE_ROOT / record.relative_path,
            destination,
        )
    tampered = tmp_path / records[0].relative_path
    content = tampered.read_text(encoding="utf-8")
    changed = content.replace(
        '"adaptive_beats_uniform":true',
        '"adaptive_beats_uniform":false',
        1,
    )
    assert changed != content
    tampered.write_text(changed, encoding="utf-8")
    with pytest.raises(
        AirfoilV10MultiOptionInputError,
        match="result file SHA-256 changed",
    ):
        authenticate_airfoil_v10_promotion_evidence(tmp_path)
    other_evidence = authenticate_airfoil_v10_other_card_promotion_evidence()
    assert other_evidence is not None


def test_airfoil_v10_gpt_record_is_existing_azure_xhigh_profile_only() -> None:
    profile = GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE
    config = build_airfoil_v10_gpt_openrouter_config()
    record = airfoil_v10_gpt_profile_config_record()

    assert record["profile_id"] == profile.profile_id == "gpt-5.6-sol-azure-xhigh"
    assert record["requested_model"] == profile.model_alias == "openai/gpt-5.6-sol"
    assert record["canonical_model"] == (
        profile.canonical_model
    ) == "openai/gpt-5.6-sol-20260709"
    assert record["provider_options"] == {
        "only": ["azure"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert record["reasoning"] == {"effort": "xhigh"}
    assert record["transport"]["reasoning"] == {"effort": "xhigh"}
    assert record["transport"]["stream_liveness"]["absolute_timeout_ns"] == (
        600_000_000_000
    )
    assert config.reasoning_config is profile.reasoning_config
    assert config.reasoning_config.to_model_setting() == {"effort": "xhigh"}
    assert record["temperature"] is None
    assert record["max_output_tokens"] == record["max_reasoning_tokens"] == 128_000
    assert record["artificial_output_cap"] is False
    assert _forbidden_reasoning_key_paths(record) == ()
    assert AIRFOIL_V10_GPT_PROFILE_CONFIG_SHA256 == (
        "31b99d24171c10cb1faa799cbcb44a8427348d9c859c2723323e2aed8fd3eee4"
    )

    # The input adapter must not pull in the unfinished planner implementation.
    adapter_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "benchmarks"
        / "engibench_airfoil"
        / "v10_multi_option_inputs.py"
    )
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))
    imported = tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    assert not any("multi_option_evolution" in name for name in imported)
