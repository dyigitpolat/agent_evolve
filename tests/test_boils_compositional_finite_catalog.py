"""Real-catalog regression for generic radius-two finite composition."""

from __future__ import annotations

from agent_evolve.agentic import (
    BoundedCompositionalFiniteVariationCatalog,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)


def _repeated_action_parent() -> FrozenJsonObject:
    parent = freeze_json(
        {
            "sequence": [
                "dsdb",
                "fraig",
                "resub",
                "rewrite_z",
                "sopb",
                "refactor",
                "resub",
                "rewrite_z",
                "resub",
                "refactor_z",
                "resub_z",
                "blut",
                "sopb",
                "blut",
                "resub",
                "refactor",
                "rewrite",
                "rewrite_z",
                "rewrite_z",
                "refactor",
            ]
        }
    )
    assert type(parent) is FrozenJsonObject
    return parent


def test_boils_composition_retains_atoms_and_admits_only_exact_radius_two() -> None:
    parent = _repeated_action_parent()
    base = BoilsFiniteVariationCatalog().options(parent)
    catalog = BoundedCompositionalFiniteVariationCatalog(
        BoilsFiniteVariationCatalog(),
        max_composite_options=128,
    )

    options = catalog.options(parent)

    assert options[:200] == base
    composites = options[200:]
    # Two of the sealed v2 sample's 128 apparent pairs interact under the
    # canonical repeated-sequence diff and must be rejected, not repaired.
    assert len(composites) == 126
    parent_sequence = thaw_json(parent)["sequence"]
    assert type(parent_sequence) is list
    for option in composites:
        child_sequence = thaw_json(option.child_configuration)["sequence"]
        assert type(child_sequence) is list
        changed = tuple(
            index
            for index, (before, after) in enumerate(
                zip(parent_sequence, child_sequence, strict=True)
            )
            if before != after
        )
        assert len(changed) == 2
        metadata = dict(option.metadata)
        assert metadata["composition_radius"] == "2"
        assert metadata["left_option_id"] != metadata["right_option_id"]

    assert len({value.option_id for value in options}) == len(options)
    assert len({value.child_configuration_sha256 for value in options}) == len(
        options
    )
