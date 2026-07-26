"""Contracts for bounded workload-neutral finite-option composition."""

from __future__ import annotations

import hashlib

from agent_evolve.agentic import (
    BoundedCompositionalFiniteVariationCatalog,
    CompositionSelectionExposure,
    FiniteVariationOption,
    FrozenJsonObject,
    bind_finite_variation_catalog,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.campaign_variation_topology import (
    CampaignVariationTopology,
    CampaignVariationTopologyMode,
)


class _FixtureCatalog:
    catalog_id = "fixture_catalog"
    catalog_version = 3
    definition_sha256 = hashlib.sha256(b"fixture-catalog-v3").hexdigest()

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent_sha256 = typed_json_sha256(parent_configuration)
        parent = thaw_json(parent_configuration)
        values = []
        for coordinate in ("a", "b", "c"):
            child = dict(parent)
            child[coordinate] = 1
            values.append(
                FiniteVariationOption(
                    option_id=f"fixture.{coordinate}",
                    parent_configuration_sha256=parent_sha256,
                    child_configuration=freeze_json(child),
                    family=f"edit_{coordinate}",
                    description=f"Set coordinate {coordinate} to one.",
                )
            )
        # This option has the same patch path as fixture.a and must never be
        # composed with it, even though its child and identity are distinct.
        conflicting = dict(parent)
        conflicting["a"] = 2
        values.append(
            FiniteVariationOption(
                option_id="fixture.a_two",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(conflicting),
                family="edit_a",
                description="Set coordinate a to two.",
            )
        )
        return tuple(values)


class _RepeatedSequenceCatalog:
    """Expose a path-disjoint pair whose canonical union is not replay-safe."""

    catalog_id = "repeated_sequence_fixture"
    catalog_version = 1
    definition_sha256 = hashlib.sha256(b"repeated-sequence-fixture-v1").hexdigest()

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent_sha256 = typed_json_sha256(parent_configuration)
        return (
            FiniteVariationOption(
                option_id="fixture.p0.b",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"sequence": ["b", "a", "b"]}),
                family="edit_0",
                description="Replace sequence position zero with b.",
            ),
            FiniteVariationOption(
                option_id="fixture.p2.a",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"sequence": ["a", "a", "a"]}),
                family="edit_2",
                description="Replace sequence position two with a.",
            ),
        )


def _parent() -> FrozenJsonObject:
    parent = freeze_json({"a": 0, "b": 0, "c": 0})
    assert type(parent) is FrozenJsonObject
    return parent


def test_compositional_catalog_retains_base_and_materializes_disjoint_unions() -> None:
    catalog = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=16,
    )
    options = catalog.options(_parent())

    assert tuple(value.option_id for value in options[:4]) == (
        "fixture.a",
        "fixture.b",
        "fixture.c",
        "fixture.a_two",
    )
    composites = options[4:]
    assert len(composites) == 5
    assert {
        tuple(sorted(thaw_json(value.child_configuration).items()))
        for value in composites
    } == {
        (("a", 0), ("b", 1), ("c", 1)),
        (("a", 1), ("b", 0), ("c", 1)),
        (("a", 1), ("b", 1), ("c", 0)),
        (("a", 2), ("b", 0), ("c", 1)),
        (("a", 2), ("b", 1), ("c", 0)),
    }
    assert all(value.family == "composite_r2" for value in composites)
    assert all(
        dict(value.metadata)["composition_radius"] == "2" for value in composites
    )
    assert all(
        set(dict(value.metadata)[key] for key in ("left_option_id", "right_option_id"))
        != {"fixture.a", "fixture.a_two"}
        for value in composites
    )


def test_compositional_catalog_is_deterministic_identity_bound_and_capped() -> None:
    first = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=2,
    )
    second = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=2,
    )

    first_contract = bind_finite_variation_catalog(first, _parent())
    second_contract = bind_finite_variation_catalog(second, _parent())

    assert first.catalog_id == _FixtureCatalog.catalog_id
    assert first.catalog_version == _FixtureCatalog.catalog_version + 2
    assert first.definition_sha256 == second.definition_sha256
    assert len(first_contract.options) == 6
    assert first_contract == second_contract
    assert len({value.option_id for value in first_contract.options}) == 6
    assert (
        len({value.child_configuration_sha256 for value in first_contract.options}) == 6
    )


def test_hierarchical_exposure_is_explicit_bounded_and_identity_distinct() -> None:
    flat = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=3,
    )
    hierarchical = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=3,
        selection_exposure=(CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION),
        required_composite_proposals=2,
    )

    flat_contract = bind_finite_variation_catalog(flat, _parent())
    hierarchical_contract = bind_finite_variation_catalog(hierarchical, _parent())

    assert hierarchical.definition_sha256 != flat.definition_sha256
    assert hierarchical.catalog_version == flat.catalog_version + 2
    composites = tuple(
        value
        for value in hierarchical_contract.options
        if value.family == "composite_r2"
    )
    assert len(composites) == 3
    for option in composites:
        metadata = dict(option.metadata)
        assert metadata["composition_selection_exposure"] == (
            "hierarchical_ranked_union"
        )
        assert metadata["composition_required_proposals"] == "2"
        assert metadata["left_option_id"] != metadata["right_option_id"]
    assert all(
        "composition_selection_exposure" not in dict(value.metadata)
        for value in flat_contract.options
    )


def test_hierarchical_exact_k8_mix_projects_to_current_action_capacity() -> None:
    hierarchical = BoundedCompositionalFiniteVariationCatalog(
        _FixtureCatalog(),
        max_composite_options=5,
        selection_exposure=(CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION),
        required_composite_proposals=2,
    )

    contract = bind_finite_variation_catalog(hierarchical, _parent())
    composites = tuple(
        value for value in contract.options if value.family == "composite_r2"
    )

    assert len(contract.options) == 9
    assert len(composites) == 5
    assert {
        dict(value.metadata)["composition_required_proposals"] for value in composites
    } == {"4"}
    assert {
        dict(value.metadata)["composition_preferred_proposals"] for value in composites
    } == {"2"}
    assert {
        dict(value.metadata)["composition_capacity_projected"] for value in composites
    } == {"true"}


def test_apparently_disjoint_sequence_pair_must_pass_exact_replay_admission() -> None:
    parent = freeze_json({"sequence": ["a", "a", "b"]})
    assert type(parent) is FrozenJsonObject
    catalog = BoundedCompositionalFiniteVariationCatalog(
        _RepeatedSequenceCatalog(),
        max_composite_options=1,
    )

    options = catalog.options(parent)

    assert tuple(value.option_id for value in options) == (
        "fixture.p0.b",
        "fixture.p2.a",
    )
    assert all(value.family != "composite_r2" for value in options)


def test_shared_campaign_topology_switch_preserves_the_catalog_port() -> None:
    atomic = CampaignVariationTopology.from_environment({})
    hierarchical = CampaignVariationTopology.from_environment(
        {
            "AGENT_EVOLVE_VARIATION_TOPOLOGY": "hierarchical_r2",
            "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "3",
            "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": "2",
        }
    )

    assert atomic.mode is CampaignVariationTopologyMode.ATOMIC
    assert atomic.decorate(_FixtureCatalog()).options(_parent()) == (
        _FixtureCatalog().options(_parent())
    )
    decorated = hierarchical.decorate(_FixtureCatalog())
    contract = bind_finite_variation_catalog(decorated, _parent())
    composites = tuple(
        value for value in contract.options if value.family == "composite_r2"
    )
    assert len(composites) == 3
    assert hierarchical.hierarchical_composition_required_proposals == 2
    assert hierarchical.to_record()["selection_exposure"] == (
        "hierarchical_ranked_union"
    )


def test_shared_topology_preserves_legacy_boils_environment_semantics() -> None:
    flat = CampaignVariationTopology.from_environment(
        {"AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "3"}
    )
    hierarchical = CampaignVariationTopology.from_environment(
        {
            "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "3",
            "AGENT_EVOLVE_COMPOSITION_SELECTION_EXPOSURE": (
                "hierarchical_ranked_union"
            ),
        }
    )

    assert flat.mode is CampaignVariationTopologyMode.FLAT_R2
    assert flat.required_composite_proposals == 0
    assert hierarchical.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2
    assert hierarchical.required_composite_proposals == 2
