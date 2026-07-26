"""Provider- and ABC-free conformance for the BOiLS public AgentEvolve port."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from agent_evolve.agentic import (
    AgenticBenchmark,
    FiniteVariationCatalog,
    FiniteVariationSelectionDraft,
    FrozenJsonObject,
    resolve_finite_variation_selection,
    thaw_json,
)
from examples.benchmarks.boils_abc.actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
    CandidateConfig,
)
from examples.benchmarks.boils_abc.agentic_benchmark import (
    benchmark,
    finite_variation_catalog,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_DEFINITION_SHA256,
    FINITE_CATALOG_ID,
    FINITE_CATALOG_VERSION,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.variation_catalog import ACTION_FAMILIES


DEFAULT_PARENT = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}


def test_boils_exposes_an_objective_only_public_benchmark() -> None:
    assert isinstance(benchmark, AgenticBenchmark)
    assert benchmark.detailed_evaluator is None
    assert benchmark.outcome_relation is None
    assert tuple((item.name, item.goal) for item in benchmark.objectives) == (
        ("total_lut_count", "min"),
        ("total_levels", "min"),
    )
    assert isinstance(finite_variation_catalog, FiniteVariationCatalog)
    assert benchmark.finite_variation_catalog_identities == (
        (
            FINITE_CATALOG_ID,
            FINITE_CATALOG_VERSION,
            FINITE_CATALOG_DEFINITION_SHA256,
        ),
    )
    assert FINITE_CATALOG_DEFINITION_SHA256 == (
        "54a0f9a034b2842be048fb85f392da5f3c5947a7b440e354feced967c2ae3cc7"
    )


def test_finite_catalog_materializes_exact_typed_single_action_children() -> None:
    contract = benchmark.bind_finite_variation(
        FINITE_CATALOG_ID,
        DEFAULT_PARENT,
    )

    assert type(contract.parent_configuration) is FrozenJsonObject
    assert len(contract.options) == SEQUENCE_LENGTH * (len(ACTION_IDS) - 1) == 200
    assert len({option.option_id for option in contract.options}) == 200
    assert len({option.child_configuration_sha256 for option in contract.options}) == 200

    parent_sequence = tuple(DEFAULT_ACTION_SEQUENCE)
    for ordinal, option in enumerate(contract.options):
        position = ordinal // (len(ACTION_IDS) - 1)
        expected_replacements = tuple(
            action_id
            for action_id in ACTION_IDS
            if action_id != parent_sequence[position]
        )
        replacement = expected_replacements[ordinal % len(expected_replacements)]
        child = CandidateConfig.model_validate(
            thaw_json(option.child_configuration),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        changed_positions = tuple(
            index
            for index, (before, after) in enumerate(
                zip(parent_sequence, child.sequence, strict=True)
            )
            if before != after
        )

        assert changed_positions == (position,)
        assert child.sequence[position] == replacement
        assert option.option_id == f"boils_abc.p{position:02d}.{replacement}"
        assert option.family == ACTION_FAMILIES[replacement]
        metadata = dict(option.metadata)
        assert metadata["position"] == f"{position:02d}"
        assert metadata["replacement_action"] == replacement
        assert json.loads(metadata["abc_commands_json"]) == list(
            ACTION_COMMANDS[replacement]
        )


def test_finite_selection_replays_only_against_its_exact_frozen_parent() -> None:
    first = benchmark.bind_finite_variation(FINITE_CATALOG_ID, DEFAULT_PARENT)
    replay = benchmark.bind_finite_variation(FINITE_CATALOG_ID, DEFAULT_PARENT)
    assert first.identity_sha256 == replay.identity_sha256
    assert tuple(option.identity_sha256 for option in first.options) == tuple(
        option.identity_sha256 for option in replay.options
    )

    selected = first.options[0]
    draft = FiniteVariationSelectionDraft(
        option_id=selected.option_id,
        option_identity_sha256=selected.identity_sha256,
        contract_identity_sha256=first.identity_sha256,
        design_rationale="Exercise the sealed BOiLS option replay boundary.",
    )
    resolved = resolve_finite_variation_selection(replay, draft)
    assert resolved == selected

    changed_parent = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}
    changed_parent["sequence"][0] = "rewrite"
    changed_contract = BoilsFiniteVariationCatalog()
    changed_benchmark = AgenticBenchmark(
        problem=benchmark.problem,
        finite_variation_catalogs=(changed_contract,),
    )
    rebound = changed_benchmark.bind_finite_variation(
        FINITE_CATALOG_ID,
        changed_parent,
    )
    assert rebound.parent_configuration_sha256 != first.parent_configuration_sha256
    assert rebound.identity_sha256 != first.identity_sha256
    with pytest.raises(ValueError, match="different finite contract"):
        resolve_finite_variation_selection(rebound, draft)


def test_boils_agentic_adapter_uses_only_the_public_agentic_facade() -> None:
    benchmark_dir = Path(__file__).parents[1] / "examples/benchmarks/boils_abc"
    for name in (
        "agentic_benchmark.py",
        "finite_variation_catalog.py",
        "problem_def.py",
    ):
        tree = ast.parse((benchmark_dir / name).read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("agent_evolve")
        }
        assert modules == {"agent_evolve.agentic"}
