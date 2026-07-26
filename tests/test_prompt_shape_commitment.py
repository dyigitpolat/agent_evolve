"""Pure contracts for treatment-blinded prompt-shape commitments."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.policies.memory.prompt_shape import (
    DefaultEvidencePromptShapePolicyV1,
    DefaultEvidencePromptShapePolicyV2,
    DefaultEvidencePromptShapePolicyV3,
    PromptShapeCommitmentPolicy,
    PromptShapeInputs,
    with_selected_insight_count,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _inputs() -> PromptShapeInputs:
    return PromptShapeInputs(
        problem_description_sha256=_hash("problem description"),
        exact_context_hash=_hash("problem/operator/phase context"),
        parent_evidence_sha256s=(_hash("parent-a"),),
        common_ancestor_evidence_sha256=None,
        operator_kind="typed_mutation",
        operator_version=1,
        phase="adaptive_control",
        allowed_top_level=("runtime",),
        mutation_contract_sha256=_hash("mutation contract"),
        mutation_response_mode="full_configuration",
        atomic_replacement_option_sha256s=(),
        candidate_schema_sha256=_hash("candidate json schema"),
        selected_insight_count=2,
        reward_definition_hash=_hash("frozen reward"),
        max_output_tokens=768,
        temperature=0.2,
    )


def test_default_policy_is_stable_treatment_blinded_and_protocol_conformant() -> None:
    policy = DefaultEvidencePromptShapePolicyV1()
    inputs = _inputs()

    adaptive = policy.commit(inputs)
    score_shuffled = policy.commit(inputs)

    assert isinstance(policy, PromptShapeCommitmentPolicy)
    assert adaptive == score_shuffled
    assert adaptive == policy.commit(replace(inputs))
    assert len(adaptive) == 64
    assert "insight" not in inputs.to_record()
    assert set(inputs.to_record()) == {
        "problem_description_sha256",
        "exact_context_hash",
        "parent_evidence_sha256s",
        "common_ancestor_evidence_sha256",
        "operator_kind",
        "operator_version",
        "phase",
        "allowed_top_level",
        "mutation_contract_sha256",
        "mutation_response_mode",
        "atomic_replacement_option_sha256s",
        "candidate_schema_sha256",
        "selected_insight_count",
        "reward_definition_hash",
        "max_output_tokens",
        "temperature_hex",
    }


def test_rooted_candidate_path_renderer_has_a_new_authenticated_pairing() -> None:
    v1 = DefaultEvidencePromptShapePolicyV1()
    v2 = DefaultEvidencePromptShapePolicyV2()
    inputs = _inputs()

    assert v2.policy_version == 2
    assert v2.renderer_policy_version == 2
    assert v2.commit(inputs) != v1.commit(inputs)
    assert v2.commit(inputs) == v2.commit(replace(inputs))


def test_discriminating_crossover_renderer_has_a_new_authenticated_pairing() -> None:
    v2 = DefaultEvidencePromptShapePolicyV2()
    v3 = DefaultEvidencePromptShapePolicyV3()
    inputs = _inputs()

    assert v3.policy_version == 3
    assert v3.renderer_policy_version == 3
    assert v3.commit(inputs) != v2.commit(inputs)
    assert v3.commit(inputs) == v3.commit(replace(inputs))


@pytest.mark.parametrize(
    "changed",
    (
        {"problem_description_sha256": _hash("different problem")},
        {"exact_context_hash": _hash("different context")},
        {"parent_evidence_sha256s": (_hash("different parent"),)},
        {"common_ancestor_evidence_sha256": _hash("ancestor")},
        {"operator_kind": "two_parent_crossover"},
        {"operator_version": 2},
        {"phase": "different_phase"},
        {"allowed_top_level": ("backend",)},
        {"mutation_contract_sha256": _hash("different contract")},
        {"mutation_response_mode": "atomic_scalar_replacement_v1"},
        {"atomic_replacement_option_sha256s": (_hash("option"),)},
        {"finite_variation_contract_sha256": _hash("finite contract")},
        {
            "crossover_response_mode": "exact_parent_import_v1",
            "exact_parent_crossover_contract_sha256": _hash(
                "exact parent crossover contract"
            ),
            "exact_parent_import_exclusions_sha256": _hash(
                "exact parent crossover exclusions"
            ),
        },
        {"candidate_schema_sha256": _hash("different schema")},
        {"selected_insight_count": 1},
        {"reward_definition_hash": _hash("different reward")},
        {"max_output_tokens": 769},
        {"temperature": 0.3},
    ),
)
def test_every_non_treatment_or_cardinality_change_changes_commitment(changed) -> None:
    policy = DefaultEvidencePromptShapePolicyV1()
    baseline = _inputs()

    assert policy.commit(replace(baseline, **changed)) != policy.commit(baseline)


def test_cardinality_helper_cannot_carry_insight_identity_or_content() -> None:
    baseline = _inputs()
    changed = with_selected_insight_count(baseline, 1)

    assert changed.selected_insight_count == 1
    assert baseline.selected_insight_count == 2
    with pytest.raises(TypeError, match="exact PromptShapeInputs"):
        with_selected_insight_count(object(), 1)  # type: ignore[arg-type]


def test_inputs_and_policy_fail_closed_on_malformed_commitment_facts() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="candidate_schema_sha256"):
        replace(inputs, candidate_schema_sha256="not-a-hash")
    with pytest.raises(ValueError, match="non-empty"):
        replace(inputs, parent_evidence_sha256s=())
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        replace(inputs, allowed_top_level=("runtime", "runtime"))
    with pytest.raises(ValueError, match="cannot repeat"):
        replace(
            inputs,
            atomic_replacement_option_sha256s=(_hash("same"), _hash("same")),
        )
    with pytest.raises(ValueError, match="temperature"):
        replace(inputs, temperature=float("nan"))
    with pytest.raises(ValueError, match="finite_variation_contract_sha256"):
        replace(inputs, finite_variation_contract_sha256="not-a-hash")
    with pytest.raises(ValueError, match="requires a crossover contract digest"):
        replace(inputs, crossover_response_mode="exact_parent_import_v1")
    with pytest.raises(ValueError, match="exact_parent_crossover_contract_sha256"):
        replace(
            inputs,
            crossover_response_mode="exact_parent_import_v1",
            exact_parent_crossover_contract_sha256="not-a-hash",
        )
    with pytest.raises(ValueError, match="requires an exclusions digest"):
        replace(
            inputs,
            crossover_response_mode="exact_parent_import_v1",
            exact_parent_crossover_contract_sha256=_hash("exact contract"),
        )
    with pytest.raises(ValueError, match="requires exact parent import"):
        replace(
            inputs,
            exact_parent_crossover_contract_sha256=_hash("unexpected contract"),
        )
    with pytest.raises(TypeError, match="exact PromptShapeInputs"):
        DefaultEvidencePromptShapePolicyV1().commit(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported prompt-shape"):
        DefaultEvidencePromptShapePolicyV1(policy_id="custom")


def test_exact_crossover_extension_is_explicit_and_legacy_record_is_stable() -> None:
    baseline = _inputs()
    contract_sha256 = _hash("exact parent crossover contract")
    exclusions_sha256 = _hash("exact parent crossover exclusions")
    exact = replace(
        baseline,
        crossover_response_mode="exact_parent_import_v1",
        exact_parent_crossover_contract_sha256=contract_sha256,
        exact_parent_import_exclusions_sha256=exclusions_sha256,
    )

    assert "crossover_response_mode" not in baseline.to_record()
    assert "exact_parent_crossover_contract_sha256" not in baseline.to_record()
    assert exact.to_record()["crossover_response_mode"] == "exact_parent_import_v1"
    assert (
        exact.to_record()["exact_parent_crossover_contract_sha256"] == contract_sha256
    )
    assert (
        exact.to_record()["exact_parent_import_exclusions_sha256"] == exclusions_sha256
    )

    policy = DefaultEvidencePromptShapePolicyV3()
    assert policy.commit(exact) != policy.commit(baseline)
    assert policy.commit(exact) != policy.commit(
        replace(
            exact,
            exact_parent_crossover_contract_sha256=_hash("different contract"),
        )
    )
    assert policy.commit(exact) != policy.commit(
        replace(
            exact,
            exact_parent_import_exclusions_sha256=_hash("different exclusions"),
        )
    )
