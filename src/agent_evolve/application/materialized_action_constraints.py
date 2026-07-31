"""Small workload-neutral policies for the materialized-action market."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionDescriptor,
)


ZERO_MATERIALIZED_SLATE_VALUE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:zero-materialized-slate-value:v1;"
    b"residual-complementarity=zero;"
    b"workload-model-provider-branches=false"
).hexdigest()
UNIQUE_PHENOTYPE_SLATE_FEASIBILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:unique-phenotype-materialized-slate-feasibility:v1;"
    b"constraint=nonempty-and-unique-materialized-phenotype;"
    b"workload-model-provider-branches=false"
).hexdigest()


@dataclass(frozen=True, slots=True)
class ZeroMaterializedSlateValue:
    """Declare that member-level values explain all predicted slate value."""

    definition_sha256: str = ZERO_MATERIALIZED_SLATE_VALUE_DEFINITION_SHA256

    def value(
        self,
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> float:
        if (
            type(actions) is not tuple
            or not actions
            or any(
                type(action) is not MaterializedActionDescriptor
                for action in actions
            )
        ):
            raise ValueError(
                "zero slate value requires a non-empty exact action tuple"
            )
        for action in actions:
            action.__post_init__()
        return 0.0


@dataclass(frozen=True, slots=True)
class UniquePhenotypeMaterializedSlateFeasibility:
    """Forbid spending two evaluator slots on one exact phenotype."""

    definition_sha256: str = (
        UNIQUE_PHENOTYPE_SLATE_FEASIBILITY_DEFINITION_SHA256
    )

    def permits(
        self,
        actions: tuple[MaterializedActionDescriptor, ...],
    ) -> bool:
        if type(actions) is not tuple or not actions:
            return False
        if any(
            type(action) is not MaterializedActionDescriptor
            for action in actions
        ):
            raise TypeError("slate contains a foreign action")
        for action in actions:
            action.__post_init__()
        phenotypes = tuple(
            action.phenotype_identity_sha256 for action in actions
        )
        return len(phenotypes) == len(set(phenotypes))


__all__ = [
    "UNIQUE_PHENOTYPE_SLATE_FEASIBILITY_DEFINITION_SHA256",
    "ZERO_MATERIALIZED_SLATE_VALUE_DEFINITION_SHA256",
    "UniquePhenotypeMaterializedSlateFeasibility",
    "ZeroMaterializedSlateValue",
]
