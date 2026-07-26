from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.application.finite_variation_eligibility import (
    OptionPhenotypeBinding,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
)
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.agentic_benchmark import benchmark
from examples.benchmarks.boils_abc.finite_variation_catalog import FINITE_CATALOG_ID


def _contract():
    return benchmark.bind_finite_variation(
        FINITE_CATALOG_ID,
        {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
    )


def test_known_phenotype_is_excluded_without_objective_data() -> None:
    contract = _contract()
    bindings = exact_configuration_phenotype_bindings(contract)
    known = (bindings[0].phenotype_identity_sha256,)

    view = eligible_finite_variation_view(
        contract=contract,
        option_phenotypes=bindings,
        known_phenotype_sha256s=known,
    )

    assert len(view.contract.options) == len(contract.options) - 1
    assert view.receipt.known_excluded_option_ids == (bindings[0].option_id,)
    assert view.receipt.alias_excluded_option_ids == ()
    assert view.receipt.base_contract_identity_sha256 == contract.identity_sha256
    assert view.receipt.receipt_sha256 == view.receipt.receipt_sha256


def test_semantic_aliases_consume_only_one_wave_slot() -> None:
    contract = _contract()
    original = exact_configuration_phenotype_bindings(contract)
    aliased = (
        original[0],
        replace(
            original[1],
            phenotype_identity_sha256=original[0].phenotype_identity_sha256,
        ),
        *original[2:],
    )

    view = eligible_finite_variation_view(
        contract=contract,
        option_phenotypes=aliased,
        known_phenotype_sha256s=(),
    )

    assert view.receipt.alias_excluded_option_ids == (original[1].option_id,)
    assert tuple(option.option_id for option in view.contract.options[:2]) == (
        original[0].option_id,
        original[2].option_id,
    )


def test_bindings_must_cover_the_exact_contract_order_and_leave_one_option() -> None:
    contract = _contract()
    bindings = exact_configuration_phenotype_bindings(contract)
    with pytest.raises(ValueError, match="exact contract order"):
        eligible_finite_variation_view(
            contract=contract,
            option_phenotypes=tuple(reversed(bindings)),
            known_phenotype_sha256s=(),
        )
    with pytest.raises(ValueError, match="removed every"):
        eligible_finite_variation_view(
            contract=contract,
            option_phenotypes=bindings,
            known_phenotype_sha256s=tuple(
                sorted(value.phenotype_identity_sha256 for value in bindings)
            ),
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        eligible_finite_variation_view(
            contract=contract,
            option_phenotypes=bindings,
            known_phenotype_sha256s=(
                bindings[1].phenotype_identity_sha256,
                bindings[0].phenotype_identity_sha256,
            ),
        )


def test_option_binding_rejects_unsealed_identity() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        OptionPhenotypeBinding("x", "not-a-hash", "0" * 64)
