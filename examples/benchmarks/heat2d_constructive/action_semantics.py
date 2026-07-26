"""Optional outcome-blind action semantics for constructive Heat2D.

The generic structural fallback can identify changed configuration paths but
cannot know the decoder's exact projection invariant: geometry-only edits keep
the requested material fraction fixed.  This benchmark-owned glossary states
that public formulation fact without revealing an evaluated configuration,
thermal consequence, useful setting, or search result.
"""

from __future__ import annotations

from dataclasses import replace

from agent_evolve.agentic import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
    FiniteVariationContract,
    derive_action_space_semantics,
)


HEAT2D_ACTION_SEMANTICS_ID = "heat2d_constructive_action_space"
HEAT2D_ACTION_SEMANTICS_VERSION = 1


def heat2d_action_space_semantics(
    contract: FiniteVariationContract,
) -> ActionSpaceSemantics:
    """Enrich exact derived paths with public decoder coupling invariants."""

    derived = derive_action_space_semantics(contract)
    axes: list[ActionAxisSemantics] = []
    for axis in derived.axes:
        path = axis.configuration_paths[0]
        if path == "$.material_fraction":
            axes.append(
                replace(
                    axis,
                    definition=(
                        "The requested material-fraction control. Changing this "
                        "coordinate changes the exact projected material_fraction "
                        "objective; the sealed option description gives the new "
                        "requested value."
                    ),
                    independence=(
                        "An atomic action on this coordinate keeps all geometry "
                        "coordinates fixed. Its thermal_term consequence remains "
                        "unknown and must be forecast with uncertainty."
                    ),
                    excluded_interpretations=(
                        "Do not assume that changing material_fraction leaves the "
                        "thermal_term objective unchanged.",
                    ),
                )
            )
            continue
        axes.append(
            replace(
                axis,
                definition=(
                    "A constructive-geometry coordinate. The qualified decoder "
                    "reprojects every geometry-only child to the unchanged "
                    "requested material fraction."
                ),
                independence=(
                    "If a sealed action does not also touch $.material_fraction, "
                    "its exact material_fraction objective delta is zero. Its "
                    "thermal_term consequence is unknown and must be forecast "
                    "with uncertainty."
                ),
                excluded_interpretations=(
                    "Do not infer a material_fraction objective change from a "
                    "geometry-only coordinate change.",
                ),
            )
        )
    return ActionSpaceSemantics(
        semantics_id=HEAT2D_ACTION_SEMANTICS_ID,
        semantics_version=HEAT2D_ACTION_SEMANTICS_VERSION,
        catalog_identities=derived.catalog_identities,
        axes=tuple(sorted(axes, key=lambda value: value.axis_id)),
    )


__all__ = [
    "HEAT2D_ACTION_SEMANTICS_ID",
    "HEAT2D_ACTION_SEMANTICS_VERSION",
    "heat2d_action_space_semantics",
]
