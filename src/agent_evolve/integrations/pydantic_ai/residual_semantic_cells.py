"""Pydantic-AI adapter for portable, outcome-blind residual semantic cells."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.application.semantic_coverage_score_portfolio import (
    MaterializedActionSemanticCell,
    MaterializedActionSemanticCellBatch,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json

from .materialized_hierarchical_residual_expert import (
    MaterializedHierarchicalResidualActionEvidence,
    MaterializedHierarchicalResidualActionEvidencePort,
)


RESIDUAL_SEMANTIC_CELL_PROJECTION_ID = (
    "pydantic_ai_residual_semantic_cells"
)
RESIDUAL_SEMANTIC_CELL_PROJECTION_VERSION = 1
_DEFINITION_DOMAIN = (
    b"agent-evolve:pydantic-ai-residual-semantic-cells:v1\x00"
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
class PydanticAIResidualSemanticCellProjection:
    """Project forecasts to categorical direction and recursive-lineage cells.

    Unknown interval directions fall back to the sign of the provider's p50
    delta.  This retains a categorical trade-off signature without trusting
    the poorly calibrated raw magnitude as an optimization score.
    """

    initial_candidate_ids: tuple[CandidateId, ...]
    evidence_sources: tuple[
        MaterializedHierarchicalResidualActionEvidencePort,
        ...,
    ] = field(repr=False, compare=False)
    projection_id: str = RESIDUAL_SEMANTIC_CELL_PROJECTION_ID
    projection_version: int = RESIDUAL_SEMANTIC_CELL_PROJECTION_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.initial_candidate_ids) is not tuple
            or self.initial_candidate_ids
            != tuple(
                sorted(
                    set(self.initial_candidate_ids),
                    key=lambda value: value.value,
                )
            )
            or any(
                type(value) is not CandidateId
                for value in self.initial_candidate_ids
            )
        ):
            raise ValueError(
                "initial_candidate_ids must be exact, unique, and canonical"
            )
        for value in self.initial_candidate_ids:
            value.__post_init__()
        if (
            type(self.evidence_sources) is not tuple
            or not self.evidence_sources
            or any(
                not isinstance(
                    value,
                    MaterializedHierarchicalResidualActionEvidencePort,
                )
                for value in self.evidence_sources
            )
        ):
            raise TypeError(
                "evidence_sources must implement the residual evidence port"
            )
        source_ids = tuple(value.expert_id for value in self.evidence_sources)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ValueError("evidence sources must be expert-canonical")
        if (
            self.projection_id != RESIDUAL_SEMANTIC_CELL_PROJECTION_ID
            or self.projection_version
            != RESIDUAL_SEMANTIC_CELL_PROJECTION_VERSION
        ):
            raise ValueError("semantic projection identity is immutable")
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
                        "initial_candidate_ids": [
                            value.value
                            for value in self.initial_candidate_ids
                        ],
                        "evidence_source_expert_ids": list(source_ids),
                        "direction_rule": (
                            "interval_direction_else_p50_sign"
                        ),
                        "recursive_lineage_rule": (
                            "any_parent_not_in_initial_candidate_ids"
                        ),
                        "missing_semantics": (
                            "empty_direction_signature_nonrecursive_unless_"
                            "descriptor_parent_is_noninitial"
                        ),
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                )
            ).hexdigest(),
        )

    def _evidence(
        self,
        action_sha256: str,
    ) -> MaterializedHierarchicalResidualActionEvidence | None:
        require_sha256(action_sha256, "action_sha256")
        matches = tuple(
            evidence
            for source in self.evidence_sources
            if (evidence := source.evidence_for(action_sha256)) is not None
        )
        if len(matches) > 1:
            raise ValueError("semantic evidence is ambiguous across experts")
        return None if not matches else matches[0]

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionSemanticCellBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        actions = tuple(
            action for proposal in proposals for action in proposal.actions
        )
        action_by_sha256 = {
            value.action_sha256: value for value in actions
        }
        if len(action_by_sha256) != len(actions):
            raise ValueError("proposal union repeats an action identity")
        initial = set(self.initial_candidate_ids)
        cells: list[MaterializedActionSemanticCell] = []
        evidence_count = 0
        for action_sha256 in sorted(action_by_sha256):
            action = action_by_sha256[action_sha256]
            evidence = self._evidence(action_sha256)
            directions: tuple[tuple[str, str], ...] = ()
            if evidence is not None:
                evidence.__post_init__()
                if evidence.action.action_sha256 != action_sha256:
                    raise ValueError("semantic evidence joins another action")
                evidence_count += 1
                normalized: list[tuple[str, str]] = []
                for forecast in evidence.effect_predictions:
                    direction = forecast.direction
                    if direction == "unknown":
                        direction = (
                            "increase"
                            if forecast.p50_delta > 0.0
                            else "decrease"
                            if forecast.p50_delta < 0.0
                            else "unchanged"
                        )
                    normalized.append((forecast.metric_id, direction))
                directions = tuple(sorted(normalized))
            cells.append(
                MaterializedActionSemanticCell(
                    action_sha256=action_sha256,
                    direction_signature=directions,
                    recursive_lineage=bool(action.parent_ids)
                    and any(value not in initial for value in action.parent_ids),
                )
            )
        return MaterializedActionSemanticCellBatch(
            projection_id=self.projection_id,
            projection_version=self.projection_version,
            projection_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=proposal_sha256s,
            cells=tuple(cells),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "schema_version": 1,
                    "action_count": len(cells),
                    "typed_residual_evidence_count": evidence_count,
                    "missing_semantics_count": len(cells) - evidence_count,
                    "initial_candidate_count": len(initial),
                    "candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "PydanticAIResidualSemanticCellProjection",
    "RESIDUAL_SEMANTIC_CELL_PROJECTION_ID",
    "RESIDUAL_SEMANTIC_CELL_PROJECTION_VERSION",
]
