"""Front-proximity parent basis: partition, floor, identity, non-regression."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.front_proximity_parent_basis import (
    FrontProximityParentBasis,
    FrontProximityParentBasisConfig,
    parent_anchor_excesses,
)
from agent_evolve.application.residual_reachability import (
    ReachabilityCandidate,
    ResidualReachabilityBasisPolicy,
    select_residual_reachability_basis,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256

METRICS = ("a", "b")


def _point(a: float, b: float):
    return (("a", a), ("b", b))


def _candidate(
    ordinal: int,
    *,
    quality: bool = False,
    initial: bool = False,
    earned: bool = False,
    cell: str | None = None,
) -> ReachabilityCandidate:
    configuration = freeze_json({"a": ordinal, "b": 0})
    return ReachabilityCandidate(
        candidate_id=CandidateId(f"candidate_{ordinal:03d}"),
        configuration=configuration,
        phenotype_identity_sha256=typed_json_sha256(configuration),
        evaluation_ordinal=ordinal,
        structural_cell=cell or f"cell.{ordinal:03d}",
        quality_archive_member=quality,
        initial_design_member=initial,
        earned_positive_lineage=earned,
    )


def _population(count: int = 12, front_size: int = 3):
    """The measured situation: a small front, a large dominated design.

    The last ``front_size`` ordinals trace a genuine trade-off front and
    carry ``quality_archive``.  Every earlier ordinal is an
    ``initial_design`` member sitting strictly behind the front, deepest
    first, so the unrestricted policy — which admits by evaluation
    ordinal — saturates its remaining slots with the FURTHEST parents.
    That is exactly the anatomy the refinement is aimed at.
    """

    front = tuple(range(count - front_size + 1, count + 1))
    candidates = []
    anchors = {}
    for ordinal in range(1, count + 1):
        near = ordinal in front
        candidate = _candidate(ordinal, quality=near, initial=not near)
        candidates.append(candidate)
        if near:
            index = front.index(ordinal)
            anchors[candidate.candidate_id] = _point(
                0.10 + 0.10 * index, 0.50 - 0.10 * index
            )
        else:
            # Strictly dominated by one front point, by a depth that
            # SHRINKS with the ordinal, so early ordinals are furthest.
            index = ordinal % front_size
            depth = 0.01 * (count - ordinal + 1)
            anchors[candidate.candidate_id] = _point(
                0.10 + 0.10 * index + depth,
                0.50 - 0.10 * index + depth,
            )
    return tuple(candidates), anchors


def _policy(maximum_parents: int = 6) -> ResidualReachabilityBasisPolicy:
    return ResidualReachabilityBasisPolicy(
        maximum_parents=maximum_parents,
        maximum_quality_archive_parents=maximum_parents,
        maximum_initial_design_parents=maximum_parents,
        maximum_earned_lineage_parents=max(1, maximum_parents // 2),
        maximum_structural_cover_parents=max(1, maximum_parents // 2),
    )


def _mean_excess(basis, excess_by_id) -> float:
    values = [
        excess_by_id[member.candidate.candidate_id]
        for member in basis.members
    ]
    return sum(values) / len(values)


# --- the geometric quantity -------------------------------------------------


def test_front_parents_score_zero_and_dominated_parents_score_positive():
    candidates, anchors = _population(6)
    excesses = parent_anchor_excesses(candidates, anchors, METRICS)
    by_id = {value.candidate_id: value for value in excesses}
    for candidate in candidates:
        value = by_id[candidate.candidate_id]
        assert value.anchored is True
        if candidate.quality_archive_member:
            assert value.excess == pytest.approx(0.0)
        else:
            assert value.excess > 0.0


def test_unanchored_parents_take_the_population_median():
    candidates, anchors = _population(6)
    orphan = candidates[0].candidate_id
    partial = {k: v for k, v in anchors.items() if k != orphan}
    excesses = parent_anchor_excesses(candidates, partial, METRICS)
    by_id = {value.candidate_id: value for value in excesses}
    known = sorted(
        value.excess for value in excesses if value.anchored
    )
    assert by_id[orphan].anchored is False
    assert by_id[orphan].excess == pytest.approx(known[len(known) // 2])


def test_excess_requires_canonically_ordered_metric_ids():
    candidates, anchors = _population(4)
    with pytest.raises(ValueError):
        parent_anchor_excesses(candidates, anchors, ("b", "a"))


# --- the restriction is a no-op where it has nothing to say -----------------


def test_full_concentration_reproduces_the_unrestricted_basis_exactly():
    candidates, anchors = _population(12)
    policy = _policy()
    arm = FrontProximityParentBasis(
        inner_policy=policy,
        config=FrontProximityParentBasisConfig(
            proximity_concentration=1.0
        ),
    )
    assert (
        arm.select(candidates, anchors, METRICS).basis_sha256
        == select_residual_reachability_basis(
            candidates, policy
        ).basis_sha256
    )


def test_no_recorded_anchor_is_a_byte_identical_no_op():
    candidates, _anchors = _population(12)
    policy = _policy()
    arm = FrontProximityParentBasis(inner_policy=policy)
    assert (
        arm.select(candidates, {}, METRICS).basis_sha256
        == select_residual_reachability_basis(
            candidates, policy
        ).basis_sha256
    )


# --- the restriction does what it claims ------------------------------------


def test_the_basis_shifts_toward_the_front():
    candidates, anchors = _population(12)
    policy = _policy()
    excess_by_id = {
        value.candidate_id: value.excess
        for value in parent_anchor_excesses(candidates, anchors, METRICS)
    }
    unrestricted = select_residual_reachability_basis(candidates, policy)
    restricted = FrontProximityParentBasis(inner_policy=policy).select(
        candidates, anchors, METRICS
    )
    assert _mean_excess(restricted, excess_by_id) < _mean_excess(
        unrestricted, excess_by_id
    )


def test_a_tighter_concentration_is_never_further_from_the_front():
    candidates, anchors = _population(16)
    policy = _policy(8)
    excess_by_id = {
        value.candidate_id: value.excess
        for value in parent_anchor_excesses(candidates, anchors, METRICS)
    }
    means = [
        _mean_excess(
            FrontProximityParentBasis(
                inner_policy=policy,
                config=FrontProximityParentBasisConfig(
                    proximity_concentration=concentration
                ),
            ).select(candidates, anchors, METRICS),
            excess_by_id,
        )
        for concentration in (1.0, 0.75, 0.5, 0.25)
    ]
    assert means == sorted(means, reverse=True)


# --- the floor is a floor, not a ban ----------------------------------------


def test_far_front_parents_keep_a_nonzero_floor():
    candidates, anchors = _population(16)
    policy = _policy(8)
    arm = FrontProximityParentBasis(
        inner_policy=policy,
        config=FrontProximityParentBasisConfig(
            proximity_concentration=0.25, far_front_floor=0.25
        ),
    )
    excesses = parent_anchor_excesses(candidates, anchors, METRICS)
    _proximal, distal = arm.partition(excesses)
    basis = arm.select(candidates, anchors, METRICS)
    selected = {
        member.candidate.candidate_id for member in basis.members
    }
    assert len(selected & set(distal)) >= 1


def test_a_zero_floor_is_not_a_configuration():
    with pytest.raises(ValueError):
        FrontProximityParentBasisConfig(far_front_floor=0.0)
    with pytest.raises(ValueError):
        FrontProximityParentBasisConfig(minimum_far_front_parents=0)


def test_a_full_floor_is_not_a_configuration():
    with pytest.raises(ValueError):
        FrontProximityParentBasisConfig(far_front_floor=1.0)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, 1, "0.5"])
def test_config_rejects_an_invalid_concentration(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        FrontProximityParentBasisConfig(proximity_concentration=value)


# --- non-regression, determinism, identity ----------------------------------


def test_the_basis_is_never_smaller_than_the_unrestricted_basis():
    candidates, anchors = _population(16)
    for maximum_parents in (2, 3, 5, 8, 13):
        policy = _policy(maximum_parents)
        unrestricted = select_residual_reachability_basis(
            candidates, policy
        )
        restricted = FrontProximityParentBasis(
            inner_policy=policy
        ).select(candidates, anchors, METRICS)
        assert len(restricted.members) == len(unrestricted.members)


def test_selection_is_independent_of_the_supplied_order():
    candidates, anchors = _population(12)
    arm = FrontProximityParentBasis(inner_policy=_policy())
    forward = arm.select(candidates, anchors, METRICS)
    reverse = arm.select(tuple(reversed(candidates)), anchors, METRICS)
    assert forward.basis_sha256 == reverse.basis_sha256


def test_ties_break_on_candidate_id_not_on_input_order():
    candidates = tuple(_candidate(index, quality=True) for index in (1, 2, 3, 4))
    anchors = {value.candidate_id: _point(0.5, 0.5) for value in candidates}
    arm = FrontProximityParentBasis(
        inner_policy=_policy(2),
        config=FrontProximityParentBasisConfig(
            proximity_concentration=0.5
        ),
    )
    proximal, _distal = arm.partition(
        parent_anchor_excesses(candidates, anchors, METRICS)
    )
    assert proximal == (
        CandidateId("candidate_001"),
        CandidateId("candidate_002"),
    )


def test_every_member_carries_an_inner_admission_reason():
    candidates, anchors = _population(12)
    basis = FrontProximityParentBasis(inner_policy=_policy()).select(
        candidates, anchors, METRICS
    )
    for member in basis.members:
        assert member.admission_reasons
        for reason in member.admission_reasons:
            assert reason.value in {
                "quality_archive",
                "initial_design",
                "earned_lineage",
                "structural_cover",
                "capacity_fill",
            }


def test_definition_sha_moves_with_config_and_inner_policy():
    base = FrontProximityParentBasis(inner_policy=_policy())
    assert (
        base.definition_sha256
        != FrontProximityParentBasis(
            inner_policy=_policy(),
            config=FrontProximityParentBasisConfig(
                proximity_concentration=0.25
            ),
        ).definition_sha256
    )
    assert (
        base.definition_sha256
        != FrontProximityParentBasis(
            inner_policy=_policy(),
            config=FrontProximityParentBasisConfig(far_front_floor=0.5),
        ).definition_sha256
    )
    assert (
        base.definition_sha256
        != FrontProximityParentBasis(
            inner_policy=_policy(7)
        ).definition_sha256
    )
    assert (
        base.definition_sha256
        == FrontProximityParentBasis(inner_policy=_policy()).definition_sha256
    )


def test_the_module_code_names_no_workload_metric_model_or_provider():
    """Provenance may be cited in the docstring; CODE may not branch on it."""

    import ast
    import agent_evolve.application.front_proximity_parent_basis as module

    text = open(module.__file__, encoding="utf-8").read()
    tree = ast.parse(text)
    docstring = ast.get_docstring(tree, clean=False) or ""
    code = text.replace(docstring, "", 1).lower()
    for banned in (
        "total_levels",
        "total_lut",
        "openrouter",
        "deepseek",
        "qwen",
        "mistral",
        "boils",
        "log2",
        "multiplier",
        "circuit",
        "workload_id=",
    ):
        assert banned not in code, banned


def test_evidence_record_is_outcome_blind_and_complete():
    candidates, anchors = _population(12)
    arm = FrontProximityParentBasis(inner_policy=_policy())
    record = arm.evidence_record(candidates, anchors, METRICS)
    assert record["source_parent_count"] == 12
    assert record["anchored_parent_count"] == 12
    assert (
        record["proximal_pool_size"] + record["distal_pool_size"] == 12
    )
    assert (
        record["unreserved_proximal_slots"]
        + record["reserved_distal_slots"]
        == 6
    )
    assert record["reserved_distal_slots"] >= 1
    assert len(record["excesses"]) == 12
    assert record["definition_sha256"] == arm.definition_sha256
    digest = hashlib.sha256(
        repr(sorted(record["config"].items())).encode("ascii")
    ).hexdigest()
    assert len(digest) == 64
