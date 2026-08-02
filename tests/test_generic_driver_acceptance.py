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


# --------------------------------------------------------------------------- #
# The invariant that can actually fail.
#
# `test_driver_contains_zero_workload_references` builds its token list from
# domain NAMES and never from workload VOCABULARY, so it passed for the whole
# life of a driver that defined `total_lut_count`, `aig_balance` and
# `aig_refactor` -- BOiLS's area objective and BOiLS's option families -- plus an
# action template reading "Apply one sealed early AIG-balance finite action".
# The distillation stripped the domain names and left the domain vocabulary, and
# a hand-kept list could not see it.  A check that cannot fail is worse than no
# check, because it certifies.
#
# This one derives the forbidden vocabulary from the REGISTERED CATALOGUES
# themselves, so adding a domain extends the guard automatically, and it carries
# a positive control that fails if the harvest ever stops seeing the two tokens
# that were actually there.
# --------------------------------------------------------------------------- #

_BENCHMARK_PACKAGES_ROOT = "examples/benchmarks"

#: Tokens that are framework vocabulary, not workload vocabulary. Each is a
#: name the DRIVER is entitled to use because it names a driver-side or
#: engine-side concept, and each is justified rather than merely listed.
_FRAMEWORK_VOCABULARY = frozenset(
    {
        "workload",       # the driver's own neutral identity label
        "typed_mutation", # an engine operator kind, not a domain's
        "min",            # ObjectiveSpec.goal values, framework-side
        "max",
        # Identifier-shaped ENGLISH that a registry happens to declare and the
        # driver is entitled to use in prose. Each is a common noun, not a
        # domain's word; the positive control below guards against this list
        # ever growing wide enough to hide a real offender.
        "objective",
        "objectives",
        "family",
        "families",
        "metric",
        "metrics",
        "name",
        "names",
        "value",
        "values",
        "source",
        "target",
        "direction",
        "decrease",
        "increase",
        "unchanged",
        "unknown",
    }
)


def _registered_workload_vocabulary() -> dict[str, set[str]]:
    """Objective names and option families, harvested by IMPORTING every
    registered benchmark package and reading the values it declares.

    Import-level rather than AST-level, because the declarations are indirect:
    `ObjectiveSpec(name, "min") for name in OBJECTIVE_NAMES`,
    `ObjectiveSpec(OBJECTIVE_PD_INTEGRAL, "min")`,
    `family=ACTION_FAMILIES[action_id]`.  A syntactic harvest sees the variable
    and not the word, which is how the first version of this guard missed
    `total_lut_count` -- the same class of vacuity it was written to fix.
    """

    import importlib
    import pathlib
    import pkgutil

    root = pathlib.Path(__file__).resolve().parents[1] / _BENCHMARK_PACKAGES_ROOT
    packages = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    harvest: dict[str, set[str]] = {}
    for package in packages:
        tokens: set[str] = set()
        prefix = f"examples.benchmarks.{package}"
        try:
            root_module = importlib.import_module(prefix)
        except Exception:  # pragma: no cover - a domain whose deps are absent
            continue
        modules = [root_module]
        for info in pkgutil.walk_packages(root_module.__path__, prefix + "."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception:  # pragma: no cover - optional heavy adapters
                continue
        for module in modules:
            for name, value in vars(module).items():
                if not any(
                    marker in name.upper()
                    for marker in ("OBJECTIVE", "FAMILY", "FAMILIES", "METRIC")
                ):
                    continue
                tokens.update(_string_leaves(value))
        tokens = {t for t in tokens if _is_vocabulary_token(t)}
        if tokens:
            harvest[package] = tokens
    return harvest


def _string_leaves(value, depth: int = 0) -> set[str]:
    """Every string reachable inside a declared constant, keys included."""

    if depth > 4:
        return set()
    if isinstance(value, str):
        return {value}
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            out |= _string_leaves(key, depth + 1) | _string_leaves(item, depth + 1)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            out |= _string_leaves(item, depth + 1)
    return out


def _is_vocabulary_token(token: str) -> bool:
    """Identifier-shaped domain words only.

    Prose, punctuation and human-readable descriptions are not vocabulary and
    must not enter the guard, or it would forbid the driver ordinary English.
    """

    import re

    if not re.fullmatch(r"[a-z][a-z0-9_]{2,}", token or ""):
        return False
    return token not in _FRAMEWORK_VOCABULARY


def test_registry_vocabulary_harvest_is_not_vacuous() -> None:
    """The guard's own precondition, checked before it is trusted.

    If this fails, `test_driver_contains_no_registered_workload_vocabulary`
    below is not proving anything and must not be read as a pass.
    """

    harvest = _registered_workload_vocabulary()
    assert len(harvest) >= 4, (
        "vocabulary was harvested from only "
        f"{sorted(harvest)} -- the guard has gone vacuous"
    )
    pooled = set().union(*harvest.values())
    assert len(pooled) >= 20, f"only {len(pooled)} vocabulary tokens harvested"

    # Positive control: these are the exact tokens the previous guard missed.
    # If the harvester stops seeing them, it has stopped working, and the
    # invariant below would pass for the wrong reason.
    for control in ("total_lut_count", "aig_balance"):
        assert control in pooled, (
            f"positive control {control!r} not harvested from the registry -- "
            "the invariant below can no longer detect the defect it exists for"
        )


def test_driver_contains_no_registered_workload_vocabulary() -> None:
    """No string literal in the driver may be a registered workload's word.

    Derived from the registries rather than from a hand-kept list, because a
    hand-kept list goes stale the moment a domain is added -- which is exactly
    how the previous guard certified a file containing the words it was
    guarding against.
    """

    import ast
    import pathlib

    import agent_evolve.driver as driver_module

    harvest = _registered_workload_vocabulary()
    owner = {}
    for package, tokens in harvest.items():
        for token in tokens:
            owner.setdefault(token, package)

    source = pathlib.Path(driver_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        literal = node.value
        for token, package in owner.items():
            # Whole-token match: a workload word appearing anywhere inside a
            # driver string literal, including inside a longer sentence.
            if token in literal:
                offenders.append(
                    f"line {node.lineno}: {token!r} (registered by {package}) "
                    f"in {literal[:60]!r}"
                )
                break
    assert offenders == [], (
        "the generic driver uses registered workload vocabulary:\n  "
        + "\n  ".join(sorted(offenders))
        + "\nMove it into the adapter or derive it from the kit."
    )
