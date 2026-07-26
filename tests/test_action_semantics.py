from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from agent_evolve.core.action_semantics import (
    ActionAxisCoordinateSemantics,
    ActionAxisSemantics,
    ActionSpaceSemantics,
    render_action_space_semantics,
)


SHAPE_CATALOG = ("airfoil_shape", 2, "1" * 64)
TRIM_CATALOG = ("airfoil_trim", 3, "2" * 64)


def _semantics() -> ActionSpaceSemantics:
    return ActionSpaceSemantics(
        semantics_id="airfoil_action_space",
        semantics_version=1,
        catalog_identities=(SHAPE_CATALOG, TRIM_CATALOG),
        axes=(
            ActionAxisSemantics(
                axis_id="shape_coefficients",
                configuration_paths=(
                    "$.lower_coefficients",
                    "$.upper_coefficients",
                ),
                option_families=("shape_only",),
                definition=(
                    "Degree-nine Bernstein basis coefficients define one shared "
                    "two-dimensional profile."
                ),
                independence=(
                    "Upper and lower changes may be coupled by the selected shape "
                    "mode and the resulting profile is shared across operating points."
                ),
                unit="dimensionless coefficient",
                excluded_interpretations=(
                    "Coefficient indices are not spanwise wing stations.",
                ),
            ),
            ActionAxisSemantics(
                axis_id="trim_operating_points",
                configuration_paths=("$.alpha_deg",),
                option_families=("trim_only",),
                definition=(
                    "The ordered coordinates are incidence settings for three "
                    "independent evaluator operating points."
                ),
                independence=(
                    "Each component changes only its same-index operating point; "
                    "components do not broadcast."
                ),
                unit="degrees",
                coordinates=(
                    ActionAxisCoordinateSemantics(
                        0,
                        "operating point 0",
                        "First independent evaluator condition.",
                    ),
                    ActionAxisCoordinateSemantics(
                        1,
                        "operating point 1",
                        "Second independent evaluator condition.",
                    ),
                    ActionAxisCoordinateSemantics(
                        2,
                        "operating point 2",
                        "Third independent evaluator condition.",
                    ),
                ),
                excluded_interpretations=(
                    "Coordinates are not chordwise geometric stations.",
                    "Coordinates are not spanwise stations or wing twist.",
                ),
            ),
        ),
    )


def test_action_semantics_are_deterministic_hash_bound_and_prompt_safe() -> None:
    first = _semantics()
    second = _semantics()

    assert first == second
    assert first.identity == (
        "airfoil_action_space",
        1,
        first.definition_sha256,
    )
    assert len(first.definition_sha256) == 64
    assert set(first.definition_sha256) <= set("0123456789abcdef")
    assert first.declared_option_families == ("shape_only", "trim_only")
    assert first.to_record() == second.to_record()

    rendered = render_action_space_semantics(first)
    assert rendered.startswith(
        "ACTION-SPACE SEMANTICS (VERSIONED, AUTHORITATIVE)\n"
    )
    record = json.loads(rendered.splitlines()[-1])
    assert record == first.to_record()
    assert record["axes"][1]["coordinates"][2] == {
        "index": 2,
        "label": "operating point 2",
        "definition": "Third independent evaluator condition.",
    }
    assert record["axes"][1]["excluded_interpretations"] == [
        "Coordinates are not chordwise geometric stations.",
        "Coordinates are not spanwise stations or wing twist.",
    ]


def test_every_semantic_change_changes_the_definition_identity() -> None:
    original = _semantics()
    changed_axis = replace(
        original.axes[1],
        independence="All coordinates broadcast to every operating point.",
    )
    changed = replace(original, axes=(original.axes[0], changed_axis))
    assert changed.definition_sha256 != original.definition_sha256

    changed_exclusion_axis = replace(
        original.axes[1],
        excluded_interpretations=(
            "Coordinates are not chordwise geometric stations.",
            "Coordinates are not temporal stages.",
        ),
    )
    changed_exclusion = replace(
        original,
        axes=(original.axes[0], changed_exclusion_axis),
    )
    assert changed_exclusion.definition_sha256 != original.definition_sha256

    changed_catalog = replace(
        original,
        catalog_identities=(SHAPE_CATALOG, ("airfoil_trim", 3, "3" * 64)),
    )
    assert changed_catalog.definition_sha256 != original.definition_sha256


def test_catalog_binding_is_order_independent_but_identity_exact() -> None:
    semantics = _semantics()
    semantics.validate_catalog_binding(
        (TRIM_CATALOG, SHAPE_CATALOG),
        ("trim_only", "shape_only", "trim_only"),
    )

    with pytest.raises(ValueError, match="catalog identities differ"):
        semantics.validate_catalog_binding(
            (SHAPE_CATALOG, ("airfoil_trim", 3, "9" * 64)),
            ("shape_only", "trim_only"),
        )
    with pytest.raises(ValueError, match="catalog identities differ"):
        semantics.validate_catalog_binding(
            (SHAPE_CATALOG, ("airfoil_trim", 4, "2" * 64)),
            ("shape_only", "trim_only"),
        )


def test_catalog_binding_requires_complete_and_only_executable_family_coverage() -> None:
    semantics = _semantics()

    with pytest.raises(ValueError, match="uncovered executable families=routing_only"):
        semantics.validate_catalog_binding(
            (SHAPE_CATALOG, TRIM_CATALOG),
            ("routing_only", "shape_only", "trim_only"),
        )
    with pytest.raises(ValueError, match="non-executable declared families=trim_only"):
        semantics.validate_catalog_binding(
            (SHAPE_CATALOG, TRIM_CATALOG),
            ("shape_only",),
        )


def test_contract_binding_accepts_one_catalog_family_subset_but_no_foreign_values() -> None:
    semantics = _semantics()

    semantics.validate_contract_binding(SHAPE_CATALOG, ("shape_only",) * 3)
    semantics.validate_contract_binding(TRIM_CATALOG, ("trim_only",))

    with pytest.raises(ValueError, match="catalog identity is absent"):
        semantics.validate_contract_binding(
            ("airfoil_shape", 2, "f" * 64),
            ("shape_only",),
        )
    with pytest.raises(ValueError, match="families absent"):
        semantics.validate_contract_binding(
            SHAPE_CATALOG,
            ("shape_only", "foreign_family"),
        )


@pytest.mark.parametrize(
    "coordinates,match",
    [
        (
            (
                ActionAxisCoordinateSemantics(1, "point 1", "First point."),
            ),
            "contiguous and ordered from zero",
        ),
        (
            (
                ActionAxisCoordinateSemantics(0, "point", "First point."),
                ActionAxisCoordinateSemantics(1, "point", "Second point."),
            ),
            "coordinate labels must be unique",
        ),
    ],
)
def test_coordinate_semantics_require_unambiguous_order(
    coordinates: tuple[ActionAxisCoordinateSemantics, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ActionAxisSemantics(
            axis_id="axis",
            configuration_paths=("$.values",),
            option_families=("edit",),
            definition="An ordered action axis.",
            independence="Each component is independently applied.",
            coordinates=coordinates,
        )


def test_action_semantics_reject_ambiguous_or_noncanonical_declarations() -> None:
    with pytest.raises(ValueError, match="canonical JSON paths"):
        ActionAxisSemantics(
            axis_id="axis",
            configuration_paths=("alpha_deg",),
            option_families=("edit",),
            definition="An axis.",
            independence="Independent coordinates.",
        )
    with pytest.raises(ValueError, match="canonically sorted"):
        ActionAxisSemantics(
            axis_id="axis",
            configuration_paths=("$.z", "$.a"),
            option_families=("edit",),
            definition="An axis.",
            independence="Independent coordinates.",
        )
    with pytest.raises(ValueError, match="control characters"):
        ActionAxisSemantics(
            axis_id="axis",
            configuration_paths=("$.value",),
            option_families=("edit",),
            definition="Line one.\nLine two.",
            independence="Independent coordinates.",
        )
    with pytest.raises(ValueError, match="canonical catalog_id order"):
        replace(_semantics(), catalog_identities=(TRIM_CATALOG, SHAPE_CATALOG))
    with pytest.raises(ValueError, match="canonical axis_id order"):
        replace(_semantics(), axes=tuple(reversed(_semantics().axes)))


def test_two_axes_cannot_publish_conflicting_meanings_for_one_path() -> None:
    first = ActionAxisSemantics(
        axis_id="first",
        configuration_paths=("$.shared",),
        option_families=("family_a",),
        definition="First purported meaning.",
        independence="Independent.",
    )
    second = ActionAxisSemantics(
        axis_id="second",
        configuration_paths=("$.shared",),
        option_families=("family_b",),
        definition="Second purported meaning.",
        independence="Independent.",
    )
    with pytest.raises(ValueError, match="multiple action axes"):
        ActionSpaceSemantics(
            semantics_id="conflicting",
            semantics_version=1,
            catalog_identities=(("catalog", 1, "a" * 64),),
            axes=(first, second),
        )


def test_action_semantics_values_are_immutable_and_renderer_is_exact_typed() -> None:
    semantics = _semantics()
    with pytest.raises(FrozenInstanceError):
        semantics.semantics_version = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="exact ActionSpaceSemantics"):
        render_action_space_semantics(semantics.to_record())  # type: ignore[arg-type]
