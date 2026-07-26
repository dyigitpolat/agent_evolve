from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_equal,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    ExactParentSource,
    build_exact_parent_import_plan,
    derive_exact_parent_crossover_contract,
    exact_parent_import_exclusions_sha256,
    materialize_exact_parent_crossover,
    replay_exact_parent_crossover,
    resolve_exact_parent_import_for_target,
    validate_exact_parent_import_exclusions,
)


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


@pytest.fixture
def ordered_parents() -> tuple[FrozenJsonObject, FrozenJsonObject]:
    base = _object(
        {
            "airfoil": {
                "shared": "unchanged",
                "control": [0.0, {"gain": 1, "mode": "base"}],
            },
            "budget": 10,
            "topology": {"old": [1, 2]},
        }
    )
    donor = _object(
        {
            "airfoil": {
                "shared": "unchanged",
                "control": [-0.0, {"gain": 2, "mode": "donor"}],
            },
            "budget": 20,
            "topology": {"new": [3, 4]},
        }
    )
    return base, donor


def test_contract_recurses_same_shape_and_uses_containing_topology_locus(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents

    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)

    assert [locus.locus_id for locus in contract.loci] == [
        "locus_0001",
        "locus_0002",
        "locus_0003",
        "locus_0004",
        "locus_0005",
    ]
    assert {locus.path_text for locus in contract.loci} == {
        '$["budget"]',
        '$["topology"]',
        '$["airfoil"]["control"][0]',
        '$["airfoil"]["control"][1]["gain"]',
        '$["airfoil"]["control"][1]["mode"]',
    }
    topology = next(
        locus for locus in contract.loci if locus.path_text == '$["topology"]'
    )
    assert topology.base_value_sha256 != topology.donor_value_sha256
    assert len(contract.contract_sha256) == 64
    assert all("shared" not in locus.path_text for locus in contract.loci)


def test_contract_path_text_is_unambiguous_for_adversarial_keys() -> None:
    base = _object({'a.b[0]"': {"x": 1}, "other": 1})
    donor = _object({'a.b[0]"': {"x": 2}, "other": 2})

    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)

    assert '$["a.b[0]\\""]["x"]' in {locus.path_text for locus in contract.loci}


@pytest.mark.parametrize(
    ("base_mismatch", "donor_mismatch"),
    [
        ({"nested": 1}, [1]),
        ([1], [1, 2]),
    ],
)
def test_type_or_array_topology_mismatch_is_one_containing_locus(
    base_mismatch: object,
    donor_mismatch: object,
) -> None:
    contract = derive_exact_parent_crossover_contract(
        base=_object({"a": 1, "b": 2, "mismatch": base_mismatch}),
        donor=_object({"a": 2, "b": 3, "mismatch": donor_mismatch}),
    )

    assert [
        locus.path_text for locus in contract.loci if "mismatch" in locus.path_text
    ] == ['$["mismatch"]']


@pytest.mark.parametrize(
    ("base", "donor"),
    [
        ({"only": 1}, {"only": 2}),
        ({"a": 1}, {"b": 1}),
        ({"same": 1}, {"same": 1}),
    ],
)
def test_contract_rejects_fewer_than_two_discriminating_loci(
    base: dict[str, object], donor: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        derive_exact_parent_crossover_contract(
            base=_object(base),
            donor=_object(donor),
        )


def test_contract_requires_already_frozen_object_roots() -> None:
    with pytest.raises(TypeError, match="exact FrozenJsonObject"):
        derive_exact_parent_crossover_contract(  # type: ignore[arg-type]
            base={"a": 1, "b": 2},
            donor=_object({"a": 2, "b": 3}),
        )


@pytest.mark.parametrize("max_loci", [True, 1, 4097])
def test_contract_rejects_invalid_hard_bound(max_loci: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_loci"):
        derive_exact_parent_crossover_contract(
            base=_object({"a": 1, "b": 2}),
            donor=_object({"a": 2, "b": 3}),
            max_loci=max_loci,  # type: ignore[arg-type]
        )


def test_contract_fails_closed_when_exact_frontier_exceeds_bound() -> None:
    with pytest.raises(ValueError, match="exceeds max_loci"):
        derive_exact_parent_crossover_contract(
            base=_object({"a": 1, "b": 2, "c": 3}),
            donor=_object({"a": 2, "b": 3, "c": 4}),
            max_loci=2,
        )


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ((), "import at least one"),
        (("locus_0001", "locus_0002"), "retain at least one"),
        (("locus_0001", "locus_0001"), "unique"),
        (("locus_0002", "locus_0001"), "canonical contract order"),
        (("locus_9999",), "outside the contract"),
    ],
)
def test_plan_rejects_noncanonical_or_nonrecombinant_selections(
    ids: tuple[str, ...], message: str
) -> None:
    contract = derive_exact_parent_crossover_contract(
        base=_object({"a": 1, "b": 2}),
        donor=_object({"a": 2, "b": 3}),
    )

    with pytest.raises(ValueError, match=message):
        build_exact_parent_import_plan(contract, ids)


def test_materialization_copies_exact_donor_subtrees_and_retains_exact_base(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    by_path = {locus.path_text: locus.locus_id for locus in contract.loci}

    result = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=tuple(
            sorted(
                (
                    by_path['$["topology"]'],
                    by_path['$["airfoil"]["control"][1]["mode"]'],
                )
            )
        ),
    )

    assert thaw_json(result.configuration) == {
        "airfoil": {
            "shared": "unchanged",
            "control": [0.0, {"gain": 1, "mode": "donor"}],
        },
        "budget": 10,
        "topology": {"new": [3, 4]},
    }
    assert {value.source for value in result.attributions} == {
        ExactParentSource.BASE,
        ExactParentSource.DONOR,
    }
    assert all(
        value.source_value_sha256 == value.materialized_value_sha256
        for value in result.attributions
    )
    assert result.contract.contract_sha256 == result.receipt.contract_sha256
    assert result.plan.plan_sha256 == result.receipt.plan_sha256
    assert result.materialization_sha256 == result.receipt.materialization_sha256
    assert len(result.receipt.receipt_sha256) == 64


def test_known_target_inverse_resolution_is_linear_exact_and_replayed(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    target = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=("locus_0002", "locus_0004"),
    ).configuration

    resolved = resolve_exact_parent_import_for_target(
        base=base,
        donor=donor,
        contract=contract,
        target=target,
    )

    assert resolved == ("locus_0002", "locus_0004")
    validate_exact_parent_import_exclusions(contract, (resolved,))
    assert len(exact_parent_import_exclusions_sha256(contract, (resolved,))) == 64
    assert (
        resolve_exact_parent_import_for_target(
            base=base,
            donor=donor,
            contract=contract,
            target=base,
        )
        is None
    )
    assert (
        resolve_exact_parent_import_for_target(
            base=base,
            donor=donor,
            contract=contract,
            target=_object({**thaw_json(target), "extra": True}),  # type: ignore[arg-type]
        )
        is None
    )


def test_import_exclusions_require_canonical_valid_proper_actions(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)

    with pytest.raises(ValueError, match="canonically sorted"):
        validate_exact_parent_import_exclusions(
            contract,
            (("locus_0002",), ("locus_0001",)),
        )
    with pytest.raises(ValueError, match="retain at least one"):
        validate_exact_parent_import_exclusions(
            contract,
            (tuple(locus.locus_id for locus in contract.loci),),
        )
    two_locus_contract = derive_exact_parent_crossover_contract(
        base=_object({"a": 1, "b": 2}),
        donor=_object({"a": 2, "b": 3}),
    )
    with pytest.raises(ValueError, match="exhaust"):
        validate_exact_parent_import_exclusions(
            two_locus_contract,
            (("locus_0001",), ("locus_0002",)),
        )


def test_materialization_rejects_contract_for_different_ordered_parents(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    stale_base = _object(
        {
            **thaw_json(base),  # type: ignore[arg-type]
            "budget": 11,
        }
    )

    with pytest.raises(ValueError, match="does not match"):
        materialize_exact_parent_crossover(
            base=stale_base,
            donor=donor,
            contract=contract,
            import_locus_ids=("locus_0001",),
        )


def test_ordered_parent_roles_change_contract_identity(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents

    forward = derive_exact_parent_crossover_contract(base=base, donor=donor)
    reverse = derive_exact_parent_crossover_contract(base=donor, donor=base)

    assert forward.contract_sha256 != reverse.contract_sha256
    assert forward.base_parent_sha256 == reverse.donor_parent_sha256


def test_receipt_replays_to_exact_same_child_and_hashes(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    original = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=("locus_0002", "locus_0004"),
    )

    replayed = replay_exact_parent_crossover(
        base=base,
        donor=donor,
        receipt=original.receipt,
    )

    assert typed_json_equal(replayed.configuration, original.configuration)
    assert replayed.materialization_sha256 == original.materialization_sha256
    assert replayed.receipt.receipt_sha256 == original.receipt.receipt_sha256


def test_receipt_rejects_tampered_materialization_hash(
    ordered_parents: tuple[FrozenJsonObject, FrozenJsonObject],
) -> None:
    base, donor = ordered_parents
    contract = derive_exact_parent_crossover_contract(base=base, donor=donor)
    original = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=("locus_0001",),
    )
    with pytest.raises(ValueError, match="does not match its evidence"):
        replace(
            original.receipt,
            materialization_sha256="0" * 64,
        )


def test_signed_zero_is_an_exact_discriminating_locus() -> None:
    contract = derive_exact_parent_crossover_contract(
        base=_object({"zero": 0.0, "other": 1}),
        donor=_object({"zero": -0.0, "other": 2}),
    )

    zero = next(locus for locus in contract.loci if locus.path_text == '$["zero"]')
    assert zero.base_value_sha256 != zero.donor_value_sha256
