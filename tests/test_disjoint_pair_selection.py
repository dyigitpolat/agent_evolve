"""Focused contracts for deterministic exploit/coverage parent selection."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from agent_evolve.domain.ids import CandidateId
from agent_evolve.policies.selection.disjoint_pairs import (
    DisjointBranchFacts,
    DisjointParentPairPolicy,
    ReplayVerifiedDisjointPair,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _branch(
    label: str,
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
    labels = f"{left.candidate_id.value}-{right.candidate_id.value}"
    return ReplayVerifiedDisjointPair(
        left=left,
        right=right,
        target_configuration_sha256=_hash(target),
        materialization_receipt_sha256=_hash(f"receipt-{labels}"),
    )


def test_exploit_then_distinct_coverage_rules_and_complete_trace() -> None:
    a = _branch("a", 6.0, "area", "family_a", 9)
    b = _branch("b", 4.0, "area", "family_b", 9)
    c = _branch("c", 4.0, "depth", "family_c", 1)
    d = _branch("d", 1.0, "depth", "family_d", 0)
    e = _branch("e", 1.0, "uncertainty", "family_e", 0)
    ab = _pair(a, b, target="ab")
    ac = _pair(a, c, target="ac")
    de = _pair(d, e, target="de")

    forward = DisjointParentPairPolicy().select((ab, ac, de))
    reverse = DisjointParentPairPolicy().select((de, ac, ab))

    assert forward == reverse
    assert forward.exploit is not None
    assert forward.exploit.pair == ac
    assert forward.exploit.branch_reward_sum == 10.0
    assert forward.exploit.distinct_role_count == 2
    assert forward.coverage is not None
    assert forward.coverage.pair == de
    assert forward.coverage.distinct_family_count == 2
    assert forward.coverage.distinct_role_count == 2
    assert forward.coverage.path_family_exposure_sum == 0

    trace = forward.to_trace_record()
    assert len(trace["eligible_rows"]) == 3
    assert trace["exploit_pair_ids"] == ["candidate_a", "candidate_c"]
    assert trace["coverage_pair_ids"] == ["candidate_d", "candidate_e"]
    assert trace["decision_sha256"] == forward.decision_sha256
    by_ids = {
        (row["left"]["candidate_id"], row["right"]["candidate_id"]): row
        for row in trace["eligible_rows"]
    }
    assert by_ids[("candidate_a", "candidate_c")][
        "target_distinct_from_exploit"
    ] is False
    assert by_ids[("candidate_d", "candidate_e")][
        "target_distinct_from_exploit"
    ] is True


def test_coverage_excludes_every_pair_with_the_exploit_target_configuration() -> None:
    a = _branch("a", 5.0, "area", "family_a", 5)
    b = _branch("b", 5.0, "depth", "family_b", 5)
    c = _branch("c", 0.0, "uncertainty", "family_c", 0)
    d = _branch("d", 0.0, "coverage", "family_d", 0)
    e = _branch("e", 0.0, "coverage", "family_e", 9)
    f = _branch("f", 0.0, "coverage", "family_f", 9)
    exploit = _pair(a, b, target="shared")
    tempting_duplicate = _pair(c, d, target="shared")
    distinct = _pair(e, f, target="distinct")

    decision = DisjointParentPairPolicy().select(
        (tempting_duplicate, distinct, exploit)
    )

    assert decision.exploit is not None
    assert decision.exploit.pair == exploit
    assert decision.coverage is not None
    assert decision.coverage.pair == distinct
    assert (
        decision.coverage.pair.target_configuration_sha256
        != decision.exploit.pair.target_configuration_sha256
    )


def test_canonical_candidate_ids_are_the_final_tie_break_for_both_roles() -> None:
    a = _branch("a", 1.0, "r1", "f1", 1)
    b = _branch("b", 1.0, "r2", "f2", 1)
    c = _branch("c", 1.0, "r1", "f1", 1)
    d = _branch("d", 1.0, "r2", "f2", 1)
    e = _branch("e", 1.0, "r1", "f1", 1)
    f = _branch("f", 1.0, "r2", "f2", 1)
    ab = _pair(a, b, target="ab")
    cd = _pair(c, d, target="cd")
    ef = _pair(e, f, target="ef")

    decision = DisjointParentPairPolicy().select((ef, cd, ab))

    assert decision.exploit is not None
    assert decision.exploit.pair == ab
    assert decision.coverage is not None
    assert decision.coverage.pair == cd


def test_empty_single_and_no_distinct_target_have_typed_skips() -> None:
    policy = DisjointParentPairPolicy()
    empty = policy.select(())
    assert empty.exploit is None
    assert empty.coverage is None

    a = _branch("a", 1.0, "area", "family_a", 0)
    b = _branch("b", 1.0, "depth", "family_b", 0)
    c = _branch("c", 0.0, "uncertainty", "family_c", 0)
    d = _branch("d", 0.0, "coverage", "family_d", 0)
    ab = _pair(a, b, target="same")
    cd = _pair(c, d, target="same")
    single = policy.select((ab,))
    assert single.exploit is not None
    assert single.coverage is None
    duplicate_targets = policy.select((cd, ab))
    assert duplicate_targets.exploit is not None
    assert duplicate_targets.coverage is None


def test_pair_records_and_cross_pair_branch_facts_fail_closed() -> None:
    a = _branch("a", 1.0, "area", "family_a", 0)
    b = _branch("b", 1.0, "depth", "family_b", 0)
    c = _branch("c", 1.0, "coverage", "family_c", 0)
    ab = _pair(a, b, target="ab")
    with pytest.raises(ValueError, match="canonically ordered"):
        ReplayVerifiedDisjointPair(
            left=b,
            right=a,
            target_configuration_sha256=_hash("bad-order"),
            materialization_receipt_sha256=_hash("bad-order-receipt"),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        ReplayVerifiedDisjointPair(
            left=a,
            right=b,
            target_configuration_sha256="bad",
            materialization_receipt_sha256=_hash("receipt"),
        )
    with pytest.raises(ValueError, match="unique"):
        DisjointParentPairPolicy().select((ab, ab))

    changed_a = _branch("a", 2.0, "area", "family_a", 0)
    ac_with_changed_facts = _pair(changed_a, c, target="ac")
    with pytest.raises(ValueError, match="inconsistent branch facts"):
        DisjointParentPairPolicy().select((ab, ac_with_changed_facts))


def test_decision_is_frozen_and_rejects_forged_selections() -> None:
    a = _branch("a", 2.0, "area", "family_a", 2)
    b = _branch("b", 2.0, "depth", "family_b", 2)
    c = _branch("c", 0.0, "coverage", "family_c", 0)
    d = _branch("d", 0.0, "uncertainty", "family_d", 0)
    decision = DisjointParentPairPolicy().select(
        (_pair(a, b, target="ab"), _pair(c, d, target="cd"))
    )
    with pytest.raises(FrozenInstanceError):
        decision.eligible_rows = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="exploit_pair_ids"):
        replace(decision, exploit_pair_ids=decision.coverage_pair_ids)
    with pytest.raises(ValueError, match="canonical"):
        replace(decision, eligible_rows=tuple(reversed(decision.eligible_rows)))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"reward": float("nan")}, ValueError, "finite"),
        ({"reward": True}, TypeError, "real"),
        ({"path_family_exposure": -1}, ValueError, "non-negative"),
        ({"role": "UPPER"}, ValueError, "role"),
    ],
)
def test_branch_facts_reject_noncanonical_values(kwargs, error, match) -> None:
    arguments: dict[str, object] = {
        "candidate_id": CandidateId("candidate_bad"),
        "reward": 1.0,
        "role": "area",
        "family": "family_a",
        "path_family_exposure": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        DisjointBranchFacts(**arguments)  # type: ignore[arg-type]

