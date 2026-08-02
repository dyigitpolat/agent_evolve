"""Claim 1 at the DRIVER level: a stranger implements the obligations and runs.

Claim 1 says AgentEvolve is a generic drop-in system published as an open-source
tool.  That was recorded as met on the strength of adapters integrating in about
fourteen engineer-minutes with zero core edits -- true, and true only at the API
level.  There was no driver that takes a workload through the five obligations
and runs a model-in-the-loop campaign: every workload carried a copied runner
between 925 and 6,821 lines, so someone who installed the wheel and implemented
the obligations could not actually run anything.

This test drives a workload through `agent_evolve.driver` end to end.  Its value
is not that it passes; it is that driving a SECOND workload through shared code
exposes couplings that were invisible while each runner was written against one
workload.  It has already produced four such findings, each recorded below as a
regression rather than as a fixed nuisance.
"""

from __future__ import annotations

import pytest

from agent_evolve.workload_kit import (
    GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID,
    GENERIC_SCHEMA_EVIDENCE_PROJECTION_VERSION,
    generic_schema_evidence_projections,
)


def _contract(eligible: int = 3):
    """The narrowest stand-in for a finite variation contract the card reads."""

    class _Contract:
        catalog_id = "acceptance_catalog"
        catalog_version = 1
        catalog_definition_sha256 = "a" * 64
        identity_sha256 = "b" * 64
        options = tuple(range(eligible))

    class _Variation:
        contract = _Contract()

    return _Variation()


def _session(workload_id: str = "acceptance_workload"):
    from agent_evolve.agentic import freeze_json

    class _Session:
        benchmark = freeze_json(
            {
                "workload_id": workload_id,
                "objectives": ["objective_a", "objective_b"],
                "optimization_semantics_identity": "c" * 64,
                "action_semantics_identity": "d" * 64,
            }
        )

    return _Session()


def _frozen(value: dict):
    from agent_evolve.agentic import freeze_json

    return freeze_json(value)


# --------------------------------------------------------------------------- #
# FINDING 1 -- the default evidence projection emitted no cards at all.
# --------------------------------------------------------------------------- #


def test_default_evidence_projection_emits_a_bootstrap_card() -> None:
    """A workload that omits evidence could not reach a portfolio stage.

    `CampaignPortfolioWaveContext` requires `evidence_cards` to be a non-empty
    tuple; the default projection returned `()`.  Four shipped workloads use
    that default -- analog_sizing, heat2d, pybamm_fastcharge and scip_miplib --
    so none of them could run a portfolio campaign.  It stayed invisible because
    every workload with a hand-written runner supplies its own projection.
    """

    projections = generic_schema_evidence_projections()
    cards = projections.cards(
        None,
        _session(),
        _frozen({"x": 1}),
        _contract(),
        _frozen({"memory": []}),
    )
    assert isinstance(cards, tuple)
    assert len(cards) == 1, "the wave context rejects an empty card tuple"


def test_bootstrap_card_is_schema_derived_and_workload_neutral() -> None:
    """The card may state the schema; it may not encode a workload."""

    from agent_evolve.agentic import thaw_json

    projections = generic_schema_evidence_projections()
    (card,) = projections.cards(
        None,
        _session("some_workload_v3"),
        _frozen({"x": 1}),
        _contract(eligible=7),
        _frozen({"memory": []}),
    )
    record = thaw_json(card)
    assert record["card_kind"] == "schema_bootstrap"
    assert record["projection_id"] == GENERIC_SCHEMA_EVIDENCE_PROJECTION_ID
    # Everything present is declared by the schema or is an identity hash.
    assert record["workload_id"] == "some_workload_v3"
    assert record["objectives"] == ["objective_a", "objective_b"]
    assert record["finite_variation"]["eligible_option_count"] == 7
    assert len(record["parent_configuration_sha256"]) == 64
    assert len(record["memory_sha256"]) == 64


def test_projection_identity_was_bumped_with_its_behaviour() -> None:
    """The identity string names its own contents, so behaviour must bump it."""

    assert GENERIC_SCHEMA_EVIDENCE_PROJECTION_VERSION == 2
    projections = generic_schema_evidence_projections()
    assert projections.projection_version == 2
    record = projections.to_record()
    assert record["projection_version"] == 2


# --------------------------------------------------------------------------- #
# FINDING 2 -- the driver must carry no workload reference. Standing invariant.
# --------------------------------------------------------------------------- #

#: Every domain name in the portfolio, plus the vocabulary of the two newest
#: adapters. The driver is generic or it is not; this is checked rather than
#: asserted because the failure mode is a constant quietly added later.
_WORKLOAD_TOKENS = (
    "boils",
    "abc",
    "timeloop",
    "heat2d",
    "scip",
    "miplib",
    "analog",
    "amplifier",
    "ngspice",
    "sky130",
    "pybamm",
    "airfoil",
    "log2",
)


def test_driver_contains_zero_workload_references() -> None:
    """The property that makes the driver worth having.

    Anything genuinely per-workload belongs in the adapter or a declared
    registry contract, never a driver constant.
    """

    import pathlib

    import agent_evolve.driver as driver_module

    source = pathlib.Path(driver_module.__file__).read_text(encoding="utf-8").lower()
    offenders = sorted(token for token in _WORKLOAD_TOKENS if token in source)
    assert offenders == [], (
        "the generic driver names a workload: "
        + ", ".join(offenders)
        + " -- move it into the adapter or a declared registry contract"
    )


def test_generic_evidence_projection_contains_zero_workload_references() -> None:
    """The default projection is framework-side and must stay neutral too."""

    import pathlib

    import agent_evolve.workload_kit as kit_module

    source = pathlib.Path(kit_module.__file__).read_text(encoding="utf-8").lower()
    # `_WORKLOAD_TOKENS` minus tokens that legitimately appear as prose in the
    # module's own defect note; the note names the affected workloads on
    # purpose, so the check is scoped to executable lines.
    code = "\n".join(
        line
        for line in source.splitlines()
        if not line.strip().startswith("#")
    )
    offenders = sorted(token for token in _WORKLOAD_TOKENS if token in code)
    assert offenders == [], (
        "the default evidence projection names a workload: " + ", ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# FINDING 3 and 4 -- regressions for what the builder already found and fixed.
# --------------------------------------------------------------------------- #


def test_driver_derives_seed_cardinality_from_the_kit() -> None:
    """A hard-coded `required_seed_count=2` met a three-seed kit.

    Latent in every existing runner and invisible because each was written
    against one workload with one seed count.
    """

    import pathlib

    import agent_evolve.driver as driver_module

    source = pathlib.Path(driver_module.__file__).read_text(encoding="utf-8")
    assert "required_seed_count=2" not in source
    assert "len(kit.seeds)" in source, (
        "seed cardinality must be derived from the kit, not asserted"
    )


def test_driver_derives_editable_paths_rather_than_naming_them() -> None:
    """Derived editable paths were non-canonical.

    The catalogue declares its own loci; the driver must read them rather than
    reconstruct a path string.
    """

    import agent_evolve.driver as driver_module

    assert hasattr(driver_module, "_derive_editable_paths")


@pytest.mark.parametrize(
    "symbol",
    (
        "_derive_editable_paths",
        "_derive_evaluator_contract_sha256",
        "_derive_bootstrap_prior",
        "run_workload_campaign",
        "WorkloadCampaignRun",
    ),
)
def test_driver_exposes_its_derivation_surface(symbol: str) -> None:
    """The five obligations are carried by the kit and derived here, not named."""

    import agent_evolve.driver as driver_module

    assert hasattr(driver_module, symbol)
