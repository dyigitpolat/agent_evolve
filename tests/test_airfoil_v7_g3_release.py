"""Focused provider-free checks for the frozen Airfoil-v7 G3 release slice."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
import json
from pathlib import Path
import sys

import pytest

# Pytest's importlib mode omits the repository root even though the benchmark
# examples are intentionally in-tree rather than part of the wheel package.
_AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[1]
if str(_AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (
    GenerationFeedbackInterceptorFactory,
    GenerationPlannerFactory,
    compose_agentic_optimizer,
)
from agent_evolve.application.agentic_evolution import OperatorKind
from agent_evolve.application.g3_causal_screen import G3_SCREEN_BUDGET
from agent_evolve.application.g3_postseal_curation import (
    G3PostsealCurationInterceptor,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightOrigin,
    context_stratum_hash,
)
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.domain.ids import CandidateId
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentAssignmentRole,
    TreatmentClaimMode,
)
from agent_evolve.ports.executable_hypothesis import (
    HypothesisApplicabilityStatus,
)
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AIRFOIL_G3_MODEL_CATALOG_ID,
    build_airfoil_g3_curation_spec,
    compose_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import AirfoilV7Problem


@cache
def _prepared() -> release.AirfoilG3ReleasePreparation:
    return release.prepare_release()


class _NoCallRawAirfoil:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_raw(self, configuration):
        del configuration
        self.calls += 1
        raise AssertionError("provider-free composition must not evaluate")


class _NoCallGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("provider-free composition must not call a provider")

    async def reflect(self, request):
        del request
        raise AssertionError("provider-free composition must not call a provider")


class _SyntheticFreezeReceipt:
    def __post_init__(self) -> None:
        return None

    def to_record(self) -> dict[str, object]:
        return {"schema_version": 1, "kind": "synthetic_freeze"}


def test_prelaunch_freeze_writer_is_atomic_and_write_once(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    receipt = _SyntheticFreezeReceipt()
    first_sha256 = release.write_prelaunch_freeze_receipt(receipt, path)
    first_bytes = path.read_bytes()
    assert first_bytes == b'{"kind":"synthetic_freeze","schema_version":1}\n'
    assert len(first_sha256) == 64

    with pytest.raises(release.AirfoilG3ReleaseError, match="write-once"):
        release.write_prelaunch_freeze_receipt(receipt, path)
    assert path.read_bytes() == first_bytes
    assert not tuple(tmp_path.glob(".freeze.json.tmp-freeze-*"))


def test_hash_only_membership_build_load_and_tamper_rejection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release, "WORKSPACE_ROOT", tmp_path)
    source = tmp_path / "history"
    source.mkdir()
    candidate = release.parent_grid_candidate(1)
    frozen = freeze_json(candidate)
    explicit_configuration_hash = "a" * 64
    explicit_candidate_hash = "b" * 64
    (source / "one.json").write_text(
        json.dumps(
            {
                "nested": {"configuration": candidate},
                "configuration_sha256": explicit_configuration_hash,
                "candidate_sha256": explicit_candidate_hash,
                "outcome": {"reward": 123.0, "rank": 1},
            }
        ),
        encoding="utf-8",
    )

    membership = release.build_historical_denylist((source,))
    assert typed_json_sha256(frozen) in membership.configuration_sha256s
    assert explicit_configuration_hash in membership.configuration_sha256s
    assert candidate_sha256(candidate) in membership.candidate_sha256s
    assert explicit_candidate_hash in membership.candidate_sha256s
    phenotype = AirfoilV7PhenotypeIdentityPolicy().identify(frozen).value_sha256
    assert phenotype in membership.phenotype_value_sha256s
    assert "123" not in json.dumps(membership.to_record())

    path = tmp_path / "membership.json"
    release.write_historical_denylist(membership, path)
    assert release.load_historical_denylist(path) == membership
    tampered = membership.to_record()
    tampered["configuration_sha256s"] = ["c" * 64]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(release.AirfoilG3ReleaseError):
        release.load_historical_denylist(path)


def test_card_bank_is_exact_typed_and_code_pinned() -> None:
    bank = release.load_authenticated_trim_card_bank()
    assert bank.card_bank_sha256 == release.EXPECTED_CARD_BANK_SHA256
    assert len(bank.source_files) == 3
    assert tuple(entry.draft.recommended_option_ids[0] for entry in bank.entries) == (
        "trim.n050.p025.n050",
        "trim.p025.n025.p050",
        "trim.n050.n025.p050",
        "trim.p050.n050.n050",
    )
    assert all(
        entry.lifecycle_state is InsightLifecycleState.QUARANTINED
        and entry.origin is InsightOrigin.REFLECTION
        and entry.applicable_operator_kinds == ("mutation",)
        and entry.evidence_lineage is not None
        for entry in bank.entries
    )

    forged_source = replace(bank.source_files[0], sha256="0" * 64)
    with pytest.raises(ValueError, match="exact source-file allowlist"):
        replace(bank, source_files=(forged_source, *bank.source_files[1:]))


def test_absolute_endpoint_is_total_bounded_and_parent_independent() -> None:
    assert release.AIRFOIL_G3_ABSOLUTE_REWARD.failure_score == -2.0
    assert release.absolute_airfoil_q(
        normalized_multipoint_drag=1.0,
        normalized_lift_equality=0.0,
        valid=False,
    ) == -2.0
    assert release.absolute_airfoil_q(
        normalized_multipoint_drag=float("nan"),
        normalized_lift_equality=0.0,
        valid=True,
    ) == -2.0
    lower_violation = release.absolute_airfoil_q(
        normalized_multipoint_drag=1.0,
        normalized_lift_equality=0.1,
        valid=True,
    )
    higher_violation = release.absolute_airfoil_q(
        normalized_multipoint_drag=1.0,
        normalized_lift_equality=0.2,
        valid=True,
    )
    lower_drag = release.absolute_airfoil_q(
        normalized_multipoint_drag=0.999,
        normalized_lift_equality=0.1,
        valid=True,
    )
    assert -1.001 < higher_violation < lower_violation <= 0.001
    assert lower_drag > lower_violation


def test_compiler_preserves_source_scope_and_rejects_foreign_projection() -> None:
    prepared = _prepared()
    compiler = release.AirfoilV7TrimHypothesisCompiler()
    entry = prepared.selected_cards[0].entry
    request = release.build_hypothesis_compilation_request(
        entry=entry,
        parent=prepared.diagnostic_parent,
        contract=prepared.diagnostic_contract,
    )
    receipt = compiler.compile(request)
    assert receipt.status is HypothesisApplicabilityStatus.APPLICABLE
    assert receipt.spec is not None
    assert receipt.spec.source_operator_kinds == ("mutation",)
    assert receipt.spec.executable_operator_kinds == (
        OperatorKind.TYPED_MUTATION.value,
    )
    assert receipt.spec.context_projection_sha256 == context_stratum_hash(
        problem_id=(
            "examples.benchmarks.engibench_airfoil.v7_problem_def."
            "AirfoilV7Problem"
        ),
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase="g3_causal_screen",
    )

    rewritten_source = compiler.compile(
        replace(request, source_operator_kinds=(OperatorKind.TYPED_MUTATION.value,))
    )
    assert rewritten_source.status is HypothesisApplicabilityStatus.INAPPLICABLE
    assert "foreign_source_operator_scope" in rewritten_source.reason_codes

    foreign_executable = compiler.compile(
        replace(
            request,
            requested_operator_kind=OperatorKind.THREE_WAY_RECOMBINATION.value,
        )
    )
    assert foreign_executable.status is HypothesisApplicabilityStatus.INAPPLICABLE
    assert "foreign_executable_operator" in foreign_executable.reason_codes


def test_provider_free_release_has_exact_fresh_budget_and_control_structure() -> None:
    prepared = _prepared()
    assert (prepared.diagnostic_parent.nonce, prepared.heldout_parent.nonce) == (
        194,
        128,
    )
    assert tuple(
        value.entry.draft.recommended_option_ids[0]
        for value in prepared.selected_cards
    ) == ("trim.p025.n025.p050", "trim.p050.n050.n050")
    assert prepared.card_selection_receipt.selected_cards == prepared.selected_cards
    assert len(prepared.card_selection_receipt.eligible_ranking) == 4
    assert tuple(
        value.selection_sha256
        for value in prepared.card_selection_receipt.eligible_ranking
    ) == tuple(
        sorted(
            value.selection_sha256
            for value in prepared.card_selection_receipt.eligible_ranking
        )
    )
    assert prepared.sham_entry.draft.recommended_option_ids == (
        "trim.n050.n050.n025",
    )
    assert prepared.sham_entry.evidence_lineage is None
    assert prepared.sham_requirement.assignment_role is TreatmentAssignmentRole.SHAM_CONTROL
    assert prepared.sham_requirement.claim_mode is TreatmentClaimMode.EXACT_REQUIRED
    assert prepared.mate_option_id == "shape.camber_aft.n0030"

    physical = prepared.freshness.physical_candidates
    schedule = prepared.freshness.occurrence_schedule
    assert len(physical) == release.MAX_UNIQUE_EVALUATIONS == 11
    assert len(schedule) == release.LOGICAL_CANDIDATE_OCCURRENCES == 12
    assert len({value.configuration_sha256 for value in physical}) == 11
    assert len({value.candidate_sha256 for value in physical}) == 11
    assert len({value.phenotype_value_sha256 for value in physical}) == 11
    assert all(
        not prepared.membership.rejects(
            configuration_sha256=value.configuration_sha256,
            candidate_sha256_value=value.candidate_sha256,
            phenotype_value_sha256=value.phenotype_value_sha256,
        )
        for value in physical
    )
    assert [row for row in schedule if row[2] == "HIT"] == [
        ("G3", "P_H_REPRODUCTION", "HIT")
    ]
    assert sum(row[2] == "MISS" for row in schedule) == 11
    assert len(prepared.freshness.recombinations) == 3

    record = prepared.to_record()
    assert record["claim_boundary"] == {
        "provider_called": False,
        "credentials_read": False,
        "physical_evaluator_called": False,
        "scientific_result_eligible": False,
        "launch_authorized": False,
        "meaning": "release preparation only; no efficacy or wall-clock result",
    }
    assert record["exact_budget"]["max_logical_llm_calls"] == 6
    assert record["live_only_terminal_requirements"]["raw_receipts"] == 11
    assert (
        record["live_only_terminal_requirements"]["total_solver_point_calls"]
        == 33
    )
    assert record["preparation_source_code"]["complete_live_launch_manifest"] is False
    assert (
        record["live_only_terminal_requirements"]["complete_launch_manifest"][
            "required"
        ]
        is True
    )


def test_public_compose_accepts_exact_airfoil_runtime_factories_without_calls() -> None:
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    raw = _NoCallRawAirfoil()
    problem = AirfoilV7Problem(raw_problem=raw)
    inputs = compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=permutation,
    )
    assert isinstance(inputs, GenerationPlannerFactory)
    assert isinstance(
        inputs.feedback_interceptor_factory,
        GenerationFeedbackInterceptorFactory,
    )
    curation_spec = build_airfoil_g3_curation_spec(preparation)
    assert inputs.feedback_interceptor_factory.spec == curation_spec
    assert curation_spec.source_scope.slot_ids == ("g2_adaptive",)
    assert curation_spec.insight_contract.required_metric_ids == (
        "objective:normalized_multipoint_drag",
        "violation:normalized_lift_equality",
    )
    assert curation_spec.insight_contract.allowed_option_families == (
        "trim_only",
    )
    assert tuple(value.reference for value in inputs.active_entries) == (
        permutation.active_references
    )
    assert inputs.memory.entries_for(inputs.active_references) == inputs.active_entries
    assert inputs.memory.entries_for((inputs.neutral_entry.reference,)) == (
        inputs.neutral_entry,
    )
    assert all(
        value.lifecycle_state is InsightLifecycleState.QUARANTINED
        and value.origin is InsightOrigin.REFLECTION
        and value.evidence_lineage is not None
        for value in inputs.active_entries
    )
    assert inputs.neutral_entry.lifecycle_state is InsightLifecycleState.QUARANTINED
    assert inputs.neutral_entry.origin is InsightOrigin.MANUAL
    assert inputs.neutral_entry.evidence_lineage is None
    assert inputs.neutral_choice.option_id == "trim.n050.n050.n025"
    assert inputs.mate_choice.option_id == "shape.camber_aft.n0030"
    with pytest.raises(ValueError, match="control choices"):
        replace(inputs, neutral_choice=inputs.mate_choice)
    with pytest.raises(ValueError, match="freeze receipt SHA-256"):
        replace(inputs, freeze_receipt_sha256="not-a-sha")
    with pytest.raises(ValueError, match="init=False"):
        replace(inputs, runtime_inputs_sha256="0" * 64)

    for index, (matrix, parent) in enumerate(zip(
        inputs.prepared_hypothesis_matrices,
        (preparation.diagnostic_parent, preparation.heldout_parent),
        strict=True,
    )):
        selected_by_reference = {
            value.entry.reference: value for value in preparation.selected_cards
        }
        expected_receipts = tuple(
            (
                selected_by_reference[entry.reference].diagnostic_receipt
                if index == 0
                else selected_by_reference[entry.reference].heldout_receipt
            )
            for entry in inputs.active_entries
        )
        assert matrix.receipts == expected_receipts
        compiled = tuple(
            inputs.benchmark.compile_registered_hypothesis_treatment(
                catalog_id=AIRFOIL_G3_MODEL_CATALOG_ID,
                parent_candidate_id=CandidateId("candidate_runtime_matrix_probe"),
                parent_configuration=parent.candidate.configuration,
                entry=entry,
                requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
                context_projection_sha256=release.CONTEXT_PROJECTION_SHA256,
                endpoint_definition_sha256=release.ABSOLUTE_Q_DEFINITION_SHA256,
            )
            for entry in inputs.active_entries
        )
        matrix.validate_runtime(compiled)

    composed = compose_agentic_optimizer(
        inputs.benchmark,
        generator=_NoCallGenerator(),
        planner_factory=inputs,
        feedback_interceptor_factory=inputs.feedback_interceptor_factory,
        budget=G3_SCREEN_BUDGET,
        seed=7,
        id_factory=inputs.id_factory,
        memory=inputs.memory,
        evaluator_concurrency=1,
        max_output_tokens=384_000,
    )
    assert composed.planner.phase == release.AIRFOIL_G3_RUNTIME_PHASE
    assert isinstance(
        composed.feedback_interceptor,
        G3PostsealCurationInterceptor,
    )
    assert composed.engine.reward_binding.failure_score == -2.0
    assert composed.id_factory is inputs.id_factory
    assert composed.memory is inputs.memory
    assert raw.calls == 0
