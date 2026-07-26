from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import stat
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from examples.benchmarks.boils_abc.budgeted_v5_analysis import (
    EXPECTED_AGENTIC_MODEL_ID,
    KNOWN_LOCAL_ORACLE_V1_SUMMARY_SHA256,
    BoilsV5AnalysisError,
    BoilsV5RunAnalysisInput,
    G1PaletteSpec,
    G1SlotObservation,
    G2SlotObservation,
    ObjectiveVector,
    OracleSealExpectation,
    ProtocolCorrectionDisclosure,
    QualityOnlyExecutionContract,
    SingleEditKey,
    SlotTreatmentAssignment,
    analyze_budgeted_v5_run,
    assess_artifact_71_mechanism_gates,
    enumerate_matched_random_portfolios,
    known_local_oracle_v1_expectation,
    paired_card_contrasts,
    parse_sealed_single_edit_oracle,
    score_generation_one,
    score_generation_two,
)
from examples.development.analyze_boils_budgeted_v5 import (
    BoilsV5AnalysisCliError,
    main as analysis_cli_main,
)
from examples.development import analyze_boils_budgeted_v5 as analysis_cli


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


SYNTHETIC_POINTS = {
    SingleEditKey(0, "a"): ObjectiveVector(8, 10),
    SingleEditKey(0, "b"): ObjectiveVector(10, 8),
    SingleEditKey(0, "c"): ObjectiveVector(11, 11),
    SingleEditKey(1, "d"): ObjectiveVector(7, 10),
    SingleEditKey(1, "e"): ObjectiveVector(10, 7),
    SingleEditKey(1, "f"): ObjectiveVector(11, 11),
    SingleEditKey(2, "g"): ObjectiveVector(6, 10),
    SingleEditKey(2, "h"): ObjectiveVector(10, 6),
    SingleEditKey(2, "i"): ObjectiveVector(11, 11),
    SingleEditKey(3, "x"): ObjectiveVector(9, 9),
}
SYNTHETIC_SUMMARY_SHA256 = _digest("synthetic-oracle-summary")
SYNTHETIC_PARENT_TYPED_SHA256 = _digest("synthetic-parent-typed")
SYNTHETIC_PARENT_BOILS_SHA256 = _digest("synthetic-parent-boils")
SYNTHETIC_PALETTE = G1PaletteSpec(
    area_index=0,
    area_replacements=("a", "b", "c"),
    depth_index=1,
    depth_replacements=("d", "e", "f"),
    uncertainty_index=2,
    uncertainty_replacements=("g", "h", "i"),
    coverage_index=3,
    coverage_replacement="x",
)


def _synthetic_expectation() -> OracleSealExpectation:
    return OracleSealExpectation(
        summary_sha256=SYNTHETIC_SUMMARY_SHA256,
        parent_boils_configuration_sha256=SYNTHETIC_PARENT_BOILS_SHA256,
        parent_typed_json_configuration_sha256=SYNTHETIC_PARENT_TYPED_SHA256,
        parent_objectives=ObjectiveVector(10, 10),
        reference_point=ObjectiveVector(12, 12),
        parent_actions=((0, "p0"), (1, "p1"), (2, "p2"), (3, "p3")),
        expected_replacements=(
            (0, ("a", "b", "c")),
            (1, ("d", "e", "f")),
            (2, ("g", "h", "i")),
            (3, ("x",)),
        ),
    )


def _synthetic_summary() -> dict[str, object]:
    outcomes: list[dict[str, object]] = [
        {
            "frozen_order": 0,
            "index": None,
            "replacement": None,
            "label": "parent_c",
            "status": "succeeded",
            "valid": True,
            "cec_passed": True,
            "candidate_local_failure_status": None,
            "objectives": {
                "total_lut_count": 10,
                "total_levels": 10,
            },
            "typed_json_configuration_sha256": SYNTHETIC_PARENT_TYPED_SHA256,
            "boils_configuration_sha256": SYNTHETIC_PARENT_BOILS_SHA256,
        }
    ]
    for frozen_order, (key, point) in enumerate(
        sorted(SYNTHETIC_POINTS.items()), start=1
    ):
        outcomes.append(
            {
                "frozen_order": frozen_order,
                "index": key.index,
                "replacement": key.replacement,
                "label": f"index_{key.index}_{key.replacement}",
                "status": "succeeded",
                "valid": True,
                "cec_passed": True,
                "candidate_local_failure_status": None,
                "objectives": point.to_record(),
                "typed_json_configuration_sha256": _digest(
                    f"typed:{key.index}:{key.replacement}"
                ),
                "boils_configuration_sha256": _digest(
                    f"boils:{key.index}:{key.replacement}"
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "succeeded",
        "development_only": True,
        "hypervolume": {
            "objective_direction": "minimize_both",
            "reference_point": {
                "total_lut_count": 12,
                "total_levels": 12,
            },
            "parent_c": 4,
            "terminal_local_oracle": 21,
        },
        "outcomes_frozen_order": outcomes,
    }


def _synthetic_oracle():
    return parse_sealed_single_edit_oracle(
        _synthetic_summary(),
        source_summary_sha256=SYNTHETIC_SUMMARY_SHA256,
        expectation=_synthetic_expectation(),
    )


def _g1_observation(slot_id: str, edit: SingleEditKey) -> G1SlotObservation:
    return G1SlotObservation(
        slot_id=slot_id,
        proposal_authority="engine" if slot_id == "G1-X" else "model",
        edit=edit,
        objectives=SYNTHETIC_POINTS[edit],
        typed_json_configuration_sha256=_digest(
            f"typed:{edit.index}:{edit.replacement}"
        ),
    )


def _synthetic_run_input(*, both_real_cards_lower: bool = False):
    g1 = (
        _g1_observation("G1-A1", SingleEditKey(0, "a")),
        _g1_observation("G1-A2", SingleEditKey(0, "c")),
        _g1_observation("G1-D1", SingleEditKey(1, "f")),
        _g1_observation("G1-D2", SingleEditKey(1, "d")),
        _g1_observation("G1-U", SingleEditKey(2, "h")),
        _g1_observation("G1-X", SingleEditKey(3, "x")),
    )
    if both_real_cards_lower:
        treatment_rows = (
            SlotTreatmentAssignment("area", "G1-A1", "placebo"),
            SlotTreatmentAssignment("area", "G1-A2", "real"),
            SlotTreatmentAssignment("depth", "G1-D1", "real"),
            SlotTreatmentAssignment("depth", "G1-D2", "placebo"),
        )
    else:
        treatment_rows = (
            SlotTreatmentAssignment("area", "G1-A1", "real"),
            SlotTreatmentAssignment("area", "G1-A2", "placebo"),
            SlotTreatmentAssignment("depth", "G1-D1", "real"),
            SlotTreatmentAssignment("depth", "G1-D2", "placebo"),
        )
    return BoilsV5RunAnalysisInput(
        agentic_model_id=EXPECTED_AGENTIC_MODEL_ID,
        development_only=True,
        protocol_acceptance_passed=True,
        post_hoc_development_protocol_correction=True,
        execution_contract=QualityOnlyExecutionContract(),
        protocol_correction=ProtocolCorrectionDisclosure.frozen_v1(),
        palette_spec=SYNTHETIC_PALETTE,
        g1_slots=g1,
        treatment_assignments=treatment_rows,
        g2_slots=(
            G2SlotObservation(
                slot_id="G2-E",
                branch_slot_ids=("G1-A1", "G1-D2"),
                objectives=ObjectiveVector(5, 8),
                typed_json_configuration_sha256=_digest("g2-exploit"),
                branch_preservation_verified=True,
                provider_telemetry_present=False,
            ),
            G2SlotObservation(
                slot_id="G2-X",
                branch_slot_ids=("G1-A2", "G1-U"),
                objectives=ObjectiveVector(7, 7),
                typed_json_configuration_sha256=_digest("g2-coverage"),
                branch_preservation_verified=True,
                provider_telemetry_present=False,
            ),
        ),
    )


def test_sealed_oracle_parser_fails_closed_on_digest_parent_and_key_support() -> None:
    oracle = _synthetic_oracle()
    assert oracle.parent_objectives == ObjectiveVector(10, 10)
    assert len(oracle.entries) == 10
    assert oracle.entry(SingleEditKey(2, "g")).objectives == ObjectiveVector(6, 10)

    with pytest.raises(BoilsV5AnalysisError, match="digest differs"):
        parse_sealed_single_edit_oracle(
            _synthetic_summary(),
            source_summary_sha256=_digest("wrong"),
            expectation=_synthetic_expectation(),
        )

    wrong_parent = _synthetic_summary()
    wrong_parent["outcomes_frozen_order"][0]["objectives"][  # type: ignore[index]
        "total_levels"
    ] = 9
    with pytest.raises(BoilsV5AnalysisError, match="parent objectives changed"):
        parse_sealed_single_edit_oracle(
            wrong_parent,
            source_summary_sha256=SYNTHETIC_SUMMARY_SHA256,
            expectation=_synthetic_expectation(),
        )

    missing = _synthetic_summary()
    missing["outcomes_frozen_order"].pop()  # type: ignore[union-attr]
    with pytest.raises(BoilsV5AnalysisError, match="edit-key support differs"):
        parse_sealed_single_edit_oracle(
            missing,
            source_summary_sha256=SYNTHETIC_SUMMARY_SHA256,
            expectation=_synthetic_expectation(),
        )


def test_exact_matched_replay_enumerates_243_and_accounts_for_cache_duplicates() -> (
    None
):
    oracle = _synthetic_oracle()
    distribution = enumerate_matched_random_portfolios(oracle, SYNTHETIC_PALETTE)
    assert len(distribution.assignments) == 3**5 == 243
    assert sum(count for _, count in distribution.support) == 243
    assert Counter(item.child_cache_hit_count for item in distribution.assignments) == {
        0: 108,
        1: 108,
        2: 27,
    }
    assert Counter(
        item.child_physical_evaluation_count for item in distribution.assignments
    ) == {6: 108, 5: 108, 4: 27}
    assert distribution.mean.denominator > 0
    assert distribution.mean == Fraction(4039, 243)
    assert distribution.first_quartile_type7 == 15
    assert distribution.median_type7 == 17
    assert distribution.third_quartile_type7 == 19
    assert (
        distribution.to_record(include_assignments=False)[
            "uniform_assignment_probability"
        ]["fraction"]
        == "1/243"
    )
    # Independent integer-grid union area, deliberately not the scorer sweep.
    for assignment in distribution.assignments:
        points = (
            oracle.parent_objectives,
            *(oracle.entry(key).objectives for key in assignment.unique_edit_keys),
        )
        brute_area = sum(
            any(
                point.total_lut_count <= x and point.total_levels <= y
                for point in points
            )
            for x in range(oracle.reference_point.total_lut_count)
            for y in range(oracle.reference_point.total_levels)
        )
        assert assignment.archive_hypervolume == brute_area


def test_generation_scores_rank_pairs_g2_interactions_and_mechanism_gates() -> None:
    oracle = _synthetic_oracle()
    distribution = enumerate_matched_random_portfolios(oracle, SYNTHETIC_PALETTE)
    run_input = _synthetic_run_input()
    assert BoilsV5RunAnalysisInput.from_record(run_input.to_record()) == run_input

    g1 = score_generation_one(run_input, oracle, distribution)
    assert g1.archive_hypervolume == 19
    assert g1.unique_child_physical_evaluations == 6
    assert g1.child_cache_hits == 0
    assert (
        g1.matched_random_rank.strictly_below_count
        + g1.matched_random_rank.equal_count
        + g1.matched_random_rank.strictly_above_count
        == 243
    )
    contrasts = paired_card_contrasts(run_input, g1)
    assert [
        (item.stratum_id, item.marginal_hypervolume_difference) for item in contrasts
    ] == [
        ("area", 4),
        ("depth", -6),
    ]
    assert all(item.edit_changed for item in contrasts)

    g2 = score_generation_two(run_input, oracle, g1)
    assert g2.generation_one_hypervolume == 19
    assert g2.terminal_hypervolume > g2.generation_one_hypervolume
    assert g2.slot_scores[0].interaction_residual is not None
    assert g2.slot_scores[0].interaction_residual.to_record() == {
        "total_lut_count": 0,
        "total_levels": -2,
    }
    assert g2.slot_scores[1].interaction_residual is not None
    assert g2.slot_scores[1].interaction_residual.to_record() == {
        "total_lut_count": -4,
        "total_levels": 0,
    }
    gates = assess_artifact_71_mechanism_gates(
        run_input, distribution, g1, contrasts, g2
    )
    assert gates.g1_strictly_above_matched_random_median is True
    assert gates.model_g1_has_positive_a0_marginal_hv is True
    assert gates.g2_has_positive_g1_marginal_or_unique_front_vector is True
    assert gates.card_delivery_changes_edit_and_reward_in_at_least_one_pair is True
    assert gates.both_real_cards_lower_than_placebos is False
    assert gates.mechanisms_advance is True

    analysis = analyze_budgeted_v5_run(run_input, oracle, distribution)
    assert analysis["post_hoc_development_protocol_correction"] is True
    assert analysis["protocol_correction"] == (
        ProtocolCorrectionDisclosure.frozen_v1().to_record()
    )
    claims = analysis["mechanism_assessment"]["claim_status"]
    assert claims == {
        "evidence_class": "shared_host_quality_only",
        "outcome_aware_design": True,
        "matched_random_support_source": "computed_from_sealed_local_oracle",
        "protocol_frozen_before_live_calls": True,
        "correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt": False,
        "confirmatory_evidence": False,
        "held_out_result": False,
        "genericity_claim_authorized": False,
        "sota_claim_authorized": False,
        "timing_comparison_claim_authorized": False,
        "wall_clock_claim_authorized": False,
        "wall_clock_dominance_claim_authorized": False,
        "only_authorized_pass_consequence": (
            "freeze the same policies on an unopened circuit and compare under a "
            "full matched budget"
        ),
    }


def test_both_real_cards_losing_triggers_frozen_retirement_rule() -> None:
    oracle = _synthetic_oracle()
    distribution = enumerate_matched_random_portfolios(oracle, SYNTHETIC_PALETTE)
    run_input = _synthetic_run_input(both_real_cards_lower=True)
    g1 = score_generation_one(run_input, oracle, distribution)
    contrasts = paired_card_contrasts(run_input, g1)
    assert [item.marginal_hypervolume_difference for item in contrasts] == [-4, -6]
    g2 = score_generation_two(run_input, oracle, g1)
    gates = assess_artifact_71_mechanism_gates(
        run_input, distribution, g1, contrasts, g2
    )
    assert gates.both_real_cards_lower_than_placebos is True
    assert gates.mechanisms_advance is False
    assert gates.next_step == (
        "retire_performance_card_retrieval_keep_uncertainty_coverage_logging"
    )


def _actual_oracle_expectation() -> OracleSealExpectation:
    return known_local_oracle_v1_expectation()


def test_checked_in_oracle_and_prospective_extended_family_correction() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    summary_path = repository_root / (
        "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
        "boils_agentic_development/boils_local_oracle_v1_20260714/summary.json"
    )
    if not summary_path.is_file():
        pytest.skip("checked-in development oracle is unavailable")
    payload = summary_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == KNOWN_LOCAL_ORACLE_V1_SUMMARY_SHA256
    oracle = parse_sealed_single_edit_oracle(
        json.loads(payload),
        source_summary_sha256=digest,
        expectation=_actual_oracle_expectation(),
    )
    assert len(oracle.entries) == 40

    # The earlier outcome-blind plan made strict > median impossible because
    # its distribution's median equalled its maximum.  Retain that diagnosis
    # explicitly rather than silently changing the baseline after seeing it.
    prior_palette = G1PaletteSpec(
        area_index=7,
        area_replacements=("dsdb", "resub", "blut"),
        depth_index=1,
        depth_replacements=("balance", "sopb", "fraig"),
        uncertainty_index=12,
        uncertainty_replacements=("balance", "sopb", "refactor_z"),
        coverage_index=18,
        coverage_replacement="blut",
    )
    prior = enumerate_matched_random_portfolios(oracle, prior_palette)
    assert prior.support == ((566, 27), (605, 81), (688, 135))
    assert prior.median_type7 == max(value for value, _ in prior.support) == 688
    assert not any(
        item.archive_hypervolume > prior.median_type7 for item in prior.assignments
    )

    # Outcome-aware development correction after inspecting the sealed matched-
    # random support: require dsdb in U, then let the unchanged task-keyed policy
    # fill/order sopb and refactor_z.  It is frozen before live calls; the
    # inspected support and dsdb outcome facts do not enter the U prompt.  The
    # design is nevertheless outcome-aware, post-hoc, and non-confirmatory.
    corrected_palette = G1PaletteSpec(
        area_index=7,
        area_replacements=("dsdb", "resub", "blut"),
        depth_index=1,
        depth_replacements=("balance", "sopb", "fraig"),
        uncertainty_index=12,
        uncertainty_replacements=("sopb", "dsdb", "refactor_z"),
        coverage_index=18,
        coverage_replacement="blut",
    )
    distribution = enumerate_matched_random_portfolios(oracle, corrected_palette)
    assert len(distribution.assignments) == 243
    assert sum(count for _, count in distribution.support) == 243
    assert distribution.reference_point == ObjectiveVector(8_028, 71)
    assert distribution.support == (
        (566, 18),
        (590, 9),
        (605, 54),
        (617, 27),
        (688, 90),
        (700, 45),
    )
    assert distribution.mean == Fraction(5861, 9)
    assert distribution.first_quartile_type7 == 605
    assert distribution.median_type7 == 688
    assert distribution.third_quartile_type7 == 688
    assert (
        sum(
            item.archive_hypervolume > distribution.median_type7
            for item in distribution.assignments
        )
        == 45
    )


def test_run_input_rejects_provider_telemetry_on_engine_only_g2() -> None:
    with pytest.raises(BoilsV5AnalysisError, match="cannot have provider telemetry"):
        G2SlotObservation(
            slot_id="G2-E",
            branch_slot_ids=("G1-A1", "G1-D1"),
            objectives=ObjectiveVector(9, 9),
            typed_json_configuration_sha256=_digest("invalid-g2"),
            branch_preservation_verified=True,
            provider_telemetry_present=True,
        )


def test_run_input_fails_closed_on_coercion_correction_and_g2_prefix() -> None:
    record = _synthetic_run_input().to_record()
    coerced = copy.deepcopy(record)
    coerced["agentic_model_id"] = 7
    with pytest.raises(BoilsV5AnalysisError, match="exact string"):
        BoilsV5RunAnalysisInput.from_record(coerced)

    weakened = copy.deepcopy(record)
    weakened["protocol_correction"]["outcome_aware_design"] = False  # type: ignore[index]
    with pytest.raises(BoilsV5AnalysisError, match="claim boundary"):
        BoilsV5RunAnalysisInput.from_record(weakened)

    original = _synthetic_run_input()
    skipped_exploit = G2SlotObservation(
        slot_id="G2-E",
        branch_slot_ids=(),
        objectives=None,
        typed_json_configuration_sha256=None,
        branch_preservation_verified=False,
        provider_telemetry_present=False,
        skipped=True,
    )
    with pytest.raises(BoilsV5AnalysisError, match="exploit-first prefix"):
        dataclasses.replace(
            original,
            g2_slots=(skipped_exploit, original.g2_slots[1]),
        )


def test_oracle_parser_does_not_mutate_caller_mapping() -> None:
    summary = _synthetic_summary()
    before = copy.deepcopy(summary)
    parse_sealed_single_edit_oracle(
        summary,
        source_summary_sha256=SYNTHETIC_SUMMARY_SHA256,
        expectation=_synthetic_expectation(),
    )
    assert summary == before


def _write_json(path: Path, value: object) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(payload)
    return payload


def _write_finalized(directory: Path, files: tuple[str, ...]) -> None:
    records = {}
    for filename in files:
        payload = (directory / filename).read_bytes()
        records[filename] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    _write_json(
        directory / "finalized.json",
        {"schema_version": 1, "status": "succeeded", "files": records},
    )


def test_read_only_cli_verifies_finalizations_and_never_mutates_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    oracle_dir = tmp_path / "oracle"
    output_dir = tmp_path / "output"
    run_dir.mkdir()
    oracle_dir.mkdir()
    output_dir.mkdir()

    run_summary = {
        "schema_version": 1,
        "status": "succeeded",
        "development_only": True,
        "protocol_acceptance_passed": True,
        "post_hoc_development_protocol_correction": True,
        "protocol_correction": ProtocolCorrectionDisclosure.frozen_v1().to_record(),
        "offline_analysis_input": _synthetic_run_input().to_record(),
    }
    _write_json(run_dir / "summary.json", run_summary)
    _write_finalized(run_dir, ("summary.json",))

    oracle_payload = _write_json(oracle_dir / "summary.json", _synthetic_summary())
    _write_finalized(oracle_dir, ("summary.json",))
    expectation = dataclasses.replace(
        _synthetic_expectation(),
        summary_sha256=hashlib.sha256(oracle_payload).hexdigest(),
    )
    evidence_before = {
        path: path.read_bytes() for path in (*run_dir.iterdir(), *oracle_dir.iterdir())
    }
    arguments = (
        str(run_dir),
        str(oracle_dir / "summary.json"),
        str(oracle_dir / "finalized.json"),
    )
    assert analysis_cli_main(arguments, oracle_expectation=expectation) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_id"] == "boils_abc_budgeted_v5_offline_analysis_cli_v1"
    assert printed["analysis"]["generation_one"]["archive_hypervolume"] == 19
    assert {
        path: path.read_bytes() for path in (*run_dir.iterdir(), *oracle_dir.iterdir())
    } == evidence_before

    original_summary = (run_dir / "summary.json").read_bytes()
    (run_dir / "summary.json").write_bytes(original_summary + b" ")
    with pytest.raises(BoilsV5AnalysisCliError, match="differs from finalization"):
        analysis_cli_main(arguments, oracle_expectation=expectation)
    (run_dir / "summary.json").write_bytes(original_summary)

    output_path = output_dir / "analysis.json"
    fsync_targets: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        fsync_targets.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(analysis_cli.os, "fsync", recording_fsync)
    assert (
        analysis_cli_main(
            (*arguments, "--output", str(output_path)),
            oracle_expectation=expectation,
        )
        == 0
    )
    assert fsync_targets[-2:] == ["file", "directory"]
    assert json.loads(output_path.read_bytes()) == printed
    with pytest.raises(BoilsV5AnalysisCliError, match="refusing overwrite"):
        analysis_cli_main(
            (*arguments, "--output", str(output_path)),
            oracle_expectation=expectation,
        )
    with pytest.raises(BoilsV5AnalysisCliError, match="inside an evidence directory"):
        analysis_cli_main(
            (*arguments, "--output", str(run_dir / "analysis.json")),
            oracle_expectation=expectation,
        )


def test_read_only_cli_rejects_unaccepted_run_before_loading_oracle(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "rejected-run"
    run_dir.mkdir()
    rejected_input = dataclasses.replace(
        _synthetic_run_input(),
        protocol_acceptance_passed=False,
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "development_only": True,
            "protocol_acceptance_passed": False,
            "post_hoc_development_protocol_correction": True,
            "protocol_correction": (
                ProtocolCorrectionDisclosure.frozen_v1().to_record()
            ),
            "offline_analysis_input": rejected_input.to_record(),
        },
    )
    _write_finalized(run_dir, ("summary.json",))

    with pytest.raises(BoilsV5AnalysisCliError, match="did not pass protocol"):
        analysis_cli.analyze_finalized_v5_run(
            run_dir,
            tmp_path / "oracle-was-not-loaded.json",
            tmp_path / "oracle-finalization-was-not-loaded.json",
            oracle_expectation=_synthetic_expectation(),
        )
