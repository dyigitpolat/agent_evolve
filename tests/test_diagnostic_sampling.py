from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, typed_json_sha256
from agent_evolve.policies.selection.diagnostic_sampling import (
    HashStratifiedDiagnosticSampler,
    validate_diagnostic_action_sample,
)


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _contract() -> FiniteVariationContract:
    parent = _frozen({"x": 0})
    parent_sha256 = typed_json_sha256(parent)
    rows = (
        ("alpha.a", "alpha", 1),
        ("alpha.b", "alpha", 2),
        ("alpha.c", "alpha", 3),
        ("beta.a", "beta", 4),
        ("beta.b", "beta", 5),
        ("beta.c", "beta", 6),
        ("gamma.a", "gamma", 7),
        ("gamma.b", "gamma", 8),
    )
    return FiniteVariationContract(
        catalog_id="diagnostic_fixture",
        catalog_version=1,
        catalog_definition_sha256=hashlib.sha256(b"fixture").hexdigest(),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen({"x": value}),
                family=family,
                description=f"Set fixture coordinate to {value}.",
            )
            for option_id, family, value in rows
        ),
    )


def test_hash_stratified_sample_is_replayable_balanced_and_contract_bound() -> None:
    contract = _contract()
    policy = HashStratifiedDiagnosticSampler(
        seed=20260715,
        design_key="generic_g1_diagnostics",
    )
    first = policy.sample(contract, sample_size=6)
    replay = policy.sample(contract, sample_size=6)

    assert first == replay
    assert first.to_record() == replay.to_record()
    assert len(first.receipt_sha256) == 64
    assert [member.rank for member in first.members] == list(range(1, 7))
    counts = {
        family: sum(member.family == family for member in first.members)
        for family in {member.family for member in first.members}
    }
    assert counts == {"alpha": 2, "beta": 2, "gamma": 2}
    validate_diagnostic_action_sample(contract, first)


def test_seed_or_design_key_changes_the_precommitted_order() -> None:
    contract = _contract()
    first = HashStratifiedDiagnosticSampler(
        seed=1,
        design_key="diagnostic_a",
    ).sample(contract, sample_size=8)
    second = HashStratifiedDiagnosticSampler(
        seed=2,
        design_key="diagnostic_a",
    ).sample(contract, sample_size=8)
    third = HashStratifiedDiagnosticSampler(
        seed=1,
        design_key="diagnostic_b",
    ).sample(contract, sample_size=8)

    assert first.receipt_sha256 != second.receipt_sha256
    assert first.receipt_sha256 != third.receipt_sha256


def test_sampling_rejects_invalid_bounds_and_tampered_members() -> None:
    contract = _contract()
    policy = HashStratifiedDiagnosticSampler(seed=0, design_key="diagnostic")
    with pytest.raises(ValueError, match="positive"):
        policy.sample(contract, sample_size=0)
    with pytest.raises(ValueError, match="exceeds"):
        policy.sample(contract, sample_size=9)

    sample = policy.sample(contract, sample_size=3)
    tampered_member = replace(
        sample.members[0],
        child_configuration_sha256="0" * 64,
    )
    tampered = replace(sample, members=(tampered_member, *sample.members[1:]))
    with pytest.raises(ValueError, match="differs"):
        validate_diagnostic_action_sample(contract, tampered)
