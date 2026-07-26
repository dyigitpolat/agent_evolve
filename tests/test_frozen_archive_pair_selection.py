"""Provider-free contracts for archive-aware recombination source selection."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, replace
from itertools import combinations
from pathlib import Path

import pytest

from agent_evolve.domain.ids import CandidateId
from agent_evolve.policies.selection.disjoint_pairs import (
    DisjointBranchFacts,
    DisjointParentPairPolicy,
    ReplayVerifiedDisjointPair,
)
from agent_evolve.policies.selection.frozen_archive_pairs import (
    ArchiveAwareDisjointParentPairPolicy,
    FrozenArchiveBranchUtility,
    FrozenArchiveSourcePairUtility,
    FrozenArchiveSourceUtilityContext,
    FrozenArchiveSourceUtilityReceipt,
    ObservedSourceBranch,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _branch(
    label: str,
    *,
    reward: float,
    role: str,
    family: str,
    exposure: int,
) -> DisjointBranchFacts:
    return DisjointBranchFacts(
        candidate_id=CandidateId(f"candidate_{label}"),
        reward=reward,
        role=role,
        family=family,
        path_family_exposure=exposure,
    )


def _pair(
    left: DisjointBranchFacts,
    right: DisjointBranchFacts,
    *,
    target: str,
) -> ReplayVerifiedDisjointPair:
    assert left.candidate_id < right.candidate_id
    label = f"{left.candidate_id.value}-{right.candidate_id.value}"
    return ReplayVerifiedDisjointPair(
        left=left,
        right=right,
        target_configuration_sha256=_hash(f"target-{target}"),
        materialization_receipt_sha256=_hash(f"materialization-{label}"),
    )


def _context(*, generation: int = 1) -> FrozenArchiveSourceUtilityContext:
    return FrozenArchiveSourceUtilityContext(
        utility_id="exact_two_objective_archive_gain",
        utility_version=1,
        utility_definition_sha256=_hash("utility-definition"),
        benchmark_sha256=_hash("benchmark"),
        archive_cutoff_sha256=_hash("archive-cutoff"),
        archive_snapshot_sha256=_hash("archive-snapshot"),
        snapshot_generation=generation,
    )


def _source(label: str, rank: int) -> ObservedSourceBranch:
    return ObservedSourceBranch(
        source_rank=rank,
        candidate_id=CandidateId(f"candidate_{label}"),
        candidate_configuration_sha256=_hash(f"configuration-{label}"),
        source_outcome_sha256=_hash(f"outcome-{label}"),
    )


def _receipt(
    utilities: tuple[tuple[str, float], ...],
    *,
    exact_pair_utilities: dict[tuple[str, str], float] | None = None,
    context: FrozenArchiveSourceUtilityContext | None = None,
    generation: int = 1,
) -> FrozenArchiveSourceUtilityReceipt:
    source_by_label = {
        label: _source(label, rank)
        for rank, (label, _utility) in enumerate(utilities, start=1)
    }
    marginal_by_label = dict(utilities)
    labels = tuple(sorted(source_by_label))
    pair_values = (
        {
            pair: max(marginal_by_label[pair[0]], marginal_by_label[pair[1]])
            for pair in combinations(labels, 2)
        }
        if exact_pair_utilities is None
        else exact_pair_utilities
    )
    return FrozenArchiveSourceUtilityReceipt(
        context=_context(generation=generation) if context is None else context,
        source_wave_receipt_sha256=_hash("source-wave"),
        source_request_sha256=_hash("source-request"),
        source_decision_sha256=_hash("source-decision"),
        source_contract_sha256=_hash("source-contract"),
        source_generation=generation,
        branches=tuple(
            FrozenArchiveBranchUtility(
                source=source_by_label[label],
                marginal_utility=utility,
            )
            for rank, (label, utility) in enumerate(utilities, start=1)
        ),
        pair_utilities=tuple(
            FrozenArchiveSourcePairUtility(
                pair_ids=(
                    source_by_label[left].candidate_id,
                    source_by_label[right].candidate_id,
                ),
                exact_joint_utility=pair_values[(left, right)],
            )
            for left, right in combinations(labels, 2)
        ),
    )


def _facts():
    a = _branch("a", reward=100.0, role="rank_0001", family="family_a", exposure=9)
    b = _branch("b", reward=90.0, role="rank_0002", family="family_b", exposure=9)
    c = _branch("c", reward=1.0, role="rank_0003", family="family_c", exposure=1)
    d = _branch("d", reward=1.0, role="rank_0004", family="family_d", exposure=0)
    return a, b, c, d


def test_archive_utility_repairs_exploit_ranking_without_changing_legacy_policy() -> (
    None
):
    a, b, c, d = _facts()
    ab = _pair(a, b, target="ab")
    ac = _pair(a, c, target="ac")
    cd = _pair(c, d, target="cd")
    pairs = (ab, ac, cd)

    legacy = DisjointParentPairPolicy().select(pairs)
    repaired = ArchiveAwareDisjointParentPairPolicy().select(
        pairs,
        source_utilities=_receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0))),
    )

    assert legacy.exploit is not None
    assert legacy.exploit.pair == ab
    assert repaired.exploit is not None
    assert repaired.exploit.pair == cd
    assert repaired.exploit.exact_joint_source_utility.exact_joint_utility == 5.0
    assert repaired.exploit.pair.left.reward + repaired.exploit.pair.right.reward == 2.0
    assert repaired.coverage is not None
    assert repaired.coverage.pair != repaired.exploit.pair

    record = repaired.to_record()
    assert record["utility_scope"] == (
        "exact_joint_observed_sources_only_not_unseen_recombination_child_"
        "performance; branch_reward_sum_is_secondary_exact_tie_break_only; "
        "marginal_utility_sum_is_diagnostic_only"
    )
    assert record["source_utility_receipt_sha256"] == (
        repaired.source_utilities.receipt_sha256
    )
    assert record["exploit_pair_ids"] == ["candidate_c", "candidate_d"]
    assert record["exploit_exact_joint_source_utility_hex"] == float(5.0).hex()


def test_exact_joint_tie_uses_observed_branch_reward_before_candidate_ids() -> None:
    a = _branch("a", reward=0.0, role="rank_0001", family="family_a", exposure=0)
    b = _branch("b", reward=0.0, role="rank_0002", family="family_b", exposure=0)
    c = _branch("c", reward=3.0, role="rank_0003", family="family_c", exposure=0)
    d = _branch("d", reward=2.0, role="rank_0004", family="family_d", exposure=0)
    identifier_preferred = _pair(a, b, target="ab")
    reward_preferred = _pair(c, d, target="cd")
    labels = ("a", "b", "c", "d")
    tied_utilities = {pair: 5.0 for pair in combinations(labels, 2)}

    tied = ArchiveAwareDisjointParentPairPolicy().select(
        (identifier_preferred, reward_preferred),
        source_utilities=_receipt(
            tuple((label, 0.0) for label in labels),
            exact_pair_utilities=tied_utilities,
        ),
    )

    assert tied.exploit is not None
    assert tied.exploit.pair == reward_preferred
    record = tied.to_record()
    assert "only within an exact archive-utility tie" in record["exploit_rule"]
    selected_row = next(
        row
        for row in record["eligible_rows"]
        if row["pair_distinct_from_exploit"] is False
    )
    assert selected_row["exploit_tie_key"][:2] == [-5.0, -5.0]

    strict_primary = ArchiveAwareDisjointParentPairPolicy().select(
        (identifier_preferred, reward_preferred),
        source_utilities=_receipt(
            tuple((label, 0.0) for label in labels),
            exact_pair_utilities={
                **tied_utilities,
                ("a", "b"): 6.0,
            },
        ),
    )
    assert strict_primary.exploit is not None
    assert strict_primary.exploit.pair == identifier_preferred


def test_exact_joint_utility_avoids_overlap_double_counting_counterexample() -> None:
    a = _branch("a", reward=0.0, role="rank_0001", family="family_a", exposure=0)
    b = _branch("b", reward=0.0, role="rank_0002", family="family_b", exposure=0)
    c = _branch("c", reward=0.0, role="rank_0003", family="family_c", exposure=0)
    overlapping = _pair(a, b, target="overlapping")
    complementary = _pair(a, c, target="complementary")
    receipt = _receipt(
        (("a", 6.0), ("b", 6.0), ("c", 5.0)),
        exact_pair_utilities={
            ("a", "b"): 6.5,
            ("a", "c"): 10.0,
            ("b", "c"): 10.0,
        },
    )

    decision = ArchiveAwareDisjointParentPairPolicy().select(
        (overlapping, complementary),
        source_utilities=receipt,
    )
    rows = {row.pair_ids: row for row in decision.eligible_rows}
    overlap_row = rows[overlapping.pair_ids]
    complementary_row = rows[complementary.pair_ids]
    overlap_marginal_sum = (
        overlap_row.left_marginal_utility.marginal_utility
        + overlap_row.right_marginal_utility.marginal_utility
    )
    complementary_marginal_sum = (
        complementary_row.left_marginal_utility.marginal_utility
        + complementary_row.right_marginal_utility.marginal_utility
    )

    assert overlap_marginal_sum == 12.0
    assert complementary_marginal_sum == 11.0
    assert overlap_marginal_sum > complementary_marginal_sum
    assert (
        overlap_row.exact_joint_source_utility.exact_joint_utility
        < complementary_row.exact_joint_source_utility.exact_joint_utility
    )
    assert decision.exploit is not None
    assert decision.exploit.pair == complementary


def test_replay_is_input_order_independent_and_commits_complete_archive_context() -> (
    None
):
    a, b, c, d = _facts()
    pairs = (
        _pair(a, b, target="ab"),
        _pair(a, c, target="ac"),
        _pair(c, d, target="cd"),
    )
    receipt = _receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0)))

    forward = ArchiveAwareDisjointParentPairPolicy().select(
        pairs, source_utilities=receipt
    )
    reverse = ArchiveAwareDisjointParentPairPolicy().select(
        tuple(reversed(pairs)), source_utilities=receipt
    )

    assert forward == reverse
    assert forward.decision_sha256 == reverse.decision_sha256
    assert tuple(row.pair for row in forward.eligible_rows) == pairs
    context_record = forward.to_record()["source_utility_receipt"]["context"]
    assert context_record["utility_definition_sha256"] == (
        receipt.context.utility_definition_sha256
    )
    assert context_record["archive_cutoff_sha256"] == (
        receipt.context.archive_cutoff_sha256
    )
    assert context_record["archive_snapshot_sha256"] == (
        receipt.context.archive_snapshot_sha256
    )
    assert len(forward.to_record()["source_utility_receipt"]["branches"]) == 4
    assert len(forward.to_record()["source_utility_receipt"]["pair_utilities"]) == 6


def test_coverage_is_structural_target_distinct_and_not_utility_ranked() -> None:
    a, b, c, d = _facts()
    exploit = _pair(c, d, target="shared")
    same_target = _pair(a, b, target="shared")
    distinct_target = _pair(a, c, target="distinct")
    receipt = _receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0)))

    decision = ArchiveAwareDisjointParentPairPolicy().select(
        (same_target, distinct_target, exploit),
        source_utilities=receipt,
    )

    assert decision.exploit is not None
    assert decision.exploit.pair == exploit
    assert decision.coverage is not None
    assert decision.coverage.pair == distinct_target
    assert decision.coverage.pair_ids != decision.exploit.pair_ids
    assert (
        decision.coverage.pair.target_configuration_sha256
        != decision.exploit.pair.target_configuration_sha256
    )


def test_coverage_uses_an_alternate_pair_when_only_same_target_is_available() -> None:
    a, b, c, d = _facts()
    exploit = _pair(c, d, target="shared")
    alternate = _pair(a, b, target="shared")
    decision = ArchiveAwareDisjointParentPairPolicy().select(
        (alternate, exploit),
        source_utilities=_receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0))),
    )

    assert decision.exploit is not None
    assert decision.coverage is not None
    assert decision.exploit.pair == exploit
    assert decision.coverage.pair == alternate
    assert (
        decision.coverage.pair.target_configuration_sha256
        == decision.exploit.pair.target_configuration_sha256
    )


def test_missing_or_foreign_source_utility_fails_closed() -> None:
    a, b, c, _d = _facts()
    ab = _pair(a, b, target="ab")
    ac = _pair(a, c, target="ac")
    incomplete = _receipt((("a", 1.0), ("b", 2.0)))

    with pytest.raises(ValueError, match="foreign or unscored"):
        ArchiveAwareDisjointParentPairPolicy().select(
            (ab, ac), source_utilities=incomplete
        )

    complete = _receipt((("a", 1.0), ("b", 2.0), ("c", 3.0)))
    with pytest.raises(ValueError, match="completely enumerate"):
        replace(complete, pair_utilities=complete.pair_utilities[:-1])


def test_receipt_authentication_rejects_stale_foreign_and_tampered_sources() -> None:
    receipt = _receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0)))
    branches = tuple(value.source for value in receipt.branches)
    arguments = {
        "context": receipt.context,
        "source_wave_receipt_sha256": receipt.source_wave_receipt_sha256,
        "source_request_sha256": receipt.source_request_sha256,
        "source_decision_sha256": receipt.source_decision_sha256,
        "source_contract_sha256": receipt.source_contract_sha256,
        "source_generation": receipt.source_generation,
        "source_branches": branches,
    }
    receipt.require_exact_context(**arguments)

    with pytest.raises(ValueError, match="foreign archive context"):
        receipt.require_exact_context(
            **{
                **arguments,
                "context": replace(
                    receipt.context,
                    archive_cutoff_sha256=_hash("foreign-cutoff"),
                ),
            }
        )
    with pytest.raises(ValueError, match="stale"):
        receipt.require_exact_context(**{**arguments, "source_generation": 2})
    with pytest.raises(ValueError, match="foreign source wave"):
        receipt.require_exact_context(
            **{
                **arguments,
                "source_wave_receipt_sha256": _hash("foreign-wave"),
            }
        )
    with pytest.raises(ValueError, match="foreign or tampered branches"):
        receipt.require_exact_context(
            **{
                **arguments,
                "source_branches": (
                    replace(
                        branches[0],
                        candidate_configuration_sha256=_hash("tampered-config"),
                    ),
                    *branches[1:],
                ),
            }
        )

    with pytest.raises(ValueError, match="stale or foreign"):
        replace(
            receipt,
            context=replace(receipt.context, snapshot_generation=2),
        )


def test_utility_and_selection_tampering_changes_commitment_or_fails_revalidation() -> (
    None
):
    a, b, c, d = _facts()
    pairs = (
        _pair(a, b, target="ab"),
        _pair(a, c, target="ac"),
        _pair(c, d, target="cd"),
    )
    receipt = _receipt((("a", 0.0), ("b", 0.0), ("c", 4.0), ("d", 5.0)))
    decision = ArchiveAwareDisjointParentPairPolicy().select(
        pairs, source_utilities=receipt
    )
    changed_receipt = replace(
        receipt,
        branches=(
            replace(receipt.branches[0], marginal_utility=100.0),
            *receipt.branches[1:],
        ),
    )
    changed_pair_receipt = replace(
        receipt,
        pair_utilities=(
            replace(receipt.pair_utilities[0], exact_joint_utility=100.0),
            *receipt.pair_utilities[1:],
        ),
    )

    assert changed_receipt.receipt_sha256 != receipt.receipt_sha256
    assert changed_pair_receipt.receipt_sha256 != receipt.receipt_sha256
    with pytest.raises(ValueError, match="eligible_rows differ"):
        replace(decision, source_utilities=changed_receipt)
    with pytest.raises(ValueError, match="eligible_rows differ"):
        replace(decision, source_utilities=changed_pair_receipt)
    with pytest.raises(ValueError, match="exploit_pair_ids"):
        replace(decision, exploit_pair_ids=decision.coverage_pair_ids)
    with pytest.raises(ValueError, match="distinct_role_count"):
        replace(decision.eligible_rows[0], distinct_role_count=True)
    with pytest.raises(FrozenInstanceError):
        decision.source_utilities = changed_receipt  # type: ignore[misc]


def test_archive_aware_types_are_public_from_selection_facade() -> None:
    import agent_evolve.policies.selection as selection

    for name, expected in (
        ("ArchiveAwareDisjointParentPairPolicy", ArchiveAwareDisjointParentPairPolicy),
        ("FrozenArchiveBranchUtility", FrozenArchiveBranchUtility),
        ("FrozenArchiveSourcePairUtility", FrozenArchiveSourcePairUtility),
        ("FrozenArchiveSourceUtilityContext", FrozenArchiveSourceUtilityContext),
        ("FrozenArchiveSourceUtilityReceipt", FrozenArchiveSourceUtilityReceipt),
        ("ObservedSourceBranch", ObservedSourceBranch),
    ):
        assert getattr(selection, name) is expected


def test_archive_aware_policy_remains_provider_and_application_free() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "agent_evolve"
        / "policies"
        / "selection"
        / "frozen_archive_pairs.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    forbidden = (
        "agent_evolve.application",
        "agent_evolve.infrastructure",
        "agent_evolve.integrations",
        "pydantic",
        "pydantic_ai",
    )
    assert not any(
        imported == blocked or imported.startswith(f"{blocked}.")
        for imported in imports
        for blocked in forbidden
    ), imports
