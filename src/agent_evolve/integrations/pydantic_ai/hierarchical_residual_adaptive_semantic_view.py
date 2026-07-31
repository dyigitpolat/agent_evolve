"""Adaptive semantic cells from generic hierarchical-residual evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionDescriptor,
)
from agent_evolve.application.outcome_adaptive_residual_portfolio_evolution import (
    AdaptiveActionSemanticView,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.integrations.pydantic_ai.materialized_hierarchical_residual_expert import (
    MaterializedHierarchicalResidualActionEvidencePort,
)


HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_ID = (
    "hierarchical_residual_adaptive_semantic_view"
)
HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_VERSION = 1
_DEFINITION_DOMAIN = (
    b"agent-evolve:hierarchical-residual-adaptive-semantic-view:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class HierarchicalResidualAdaptiveSemanticView:
    """Hash portable component and forecast-direction claims into cells."""

    evidence_sources: tuple[
        MaterializedHierarchicalResidualActionEvidencePort,
        ...,
    ] = field(repr=False, compare=False)
    projection_id: str = HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_ID
    projection_version: int = (
        HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.evidence_sources) is not tuple
            or not self.evidence_sources
        ):
            raise ValueError("evidence_sources must be non-empty")
        identities: list[dict[str, object]] = []
        for source in self.evidence_sources:
            if not isinstance(
                source,
                MaterializedHierarchicalResidualActionEvidencePort,
            ):
                raise TypeError("evidence source must implement its port")
            expert_id = getattr(source, "expert_id", None)
            expert_version = getattr(source, "expert_version", None)
            definition_sha256 = getattr(
                source,
                "definition_sha256",
                None,
            )
            if type(expert_id) is not str or not expert_id:
                raise ValueError("evidence source expert_id is malformed")
            if type(expert_version) is not int or expert_version <= 0:
                raise ValueError("evidence source version is malformed")
            require_sha256(
                definition_sha256,
                "evidence source definition",
            )
            identities.append(
                {
                    "expert_id": expert_id,
                    "expert_version": expert_version,
                    "definition_sha256": definition_sha256,
                }
            )
        if [value["expert_id"] for value in identities] != sorted(
            {value["expert_id"] for value in identities}
        ):
            raise ValueError(
                "evidence sources must be unique and expert-canonical"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _DEFINITION_DOMAIN
                + _canonical_json(
                    {
                        "schema_version": 1,
                        "projection_id": self.projection_id,
                        "projection_version": self.projection_version,
                        "evidence_sources": identities,
                        "operator_group": "component_count",
                        "semantic_cells": (
                            "hashed_component_identity_and_forecast_direction"
                        ),
                        "candidate_outcomes_observed": False,
                        "workload_objective_model_provider_prompt_branches": (
                            False
                        ),
                    }
                )
            ).hexdigest(),
        )

    def view(
        self,
        action: MaterializedActionDescriptor,
    ) -> AdaptiveActionSemanticView:
        self.__post_init__()
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be exact")
        action.__post_init__()
        matches = tuple(
            value
            for source in self.evidence_sources
            if (
                value := source.evidence_for(action.action_sha256)
            )
            is not None
        )
        if len(matches) != 1:
            raise ValueError(
                "one action must have exactly one semantic evidence row"
            )
        evidence = matches[0]
        evidence.__post_init__()
        cells = {
            f"components:{len(evidence.plan.component_option_ids)}"
        }
        cells.update(
            "component:"
            + hashlib.sha256(
                value.encode("utf-8")
            ).hexdigest()[:16]
            for value in evidence.plan.component_option_ids
        )
        cells.update(
            "direction:"
            + hashlib.sha256(
                (
                    f"{value.metric_id}:"
                    + (
                        "increase"
                        if value.p50_delta > 0.0
                        else "decrease"
                        if value.p50_delta < 0.0
                        else "unchanged"
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            for value in evidence.effect_predictions
        )
        return AdaptiveActionSemanticView(
            operator_id=(
                f"components:{len(evidence.plan.component_option_ids)}"
            ),
            semantic_cell_ids=tuple(sorted(cells)),
        )


__all__ = [
    "HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_ID",
    "HIERARCHICAL_RESIDUAL_ADAPTIVE_SEMANTIC_VIEW_VERSION",
    "HierarchicalResidualAdaptiveSemanticView",
]
