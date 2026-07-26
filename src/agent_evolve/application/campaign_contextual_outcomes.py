"""Campaign adapter for parent-aware empirical outcome history.

The outcome ledger is deliberately independent from campaign execution.  This
adapter joins the two at the last safe point before a selector request is
built: it derives lineage and currently available action families from the
authenticated campaign context, performs a prior-only query, and returns the
bounded prompt projection expected by ``CampaignPortfolioContextEnricher``.

No objective name, workload, model, or provider is encoded here.  Workloads
with a stronger structural similarity notion may inject an exact relevance
resolver without replacing the retrieval or audit boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignPortfolioWaveContext,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    ContextualOutcomeQuery,
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.domain.typed_json import FrozenJsonObject


@runtime_checkable
class CampaignOutcomeRelevanceResolver(Protocol):
    """Return canonical family/path filters for one prospective wave."""

    def resolve(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]: ...


@dataclass(frozen=True, slots=True)
class AvailableFamilyOutcomeRelevance:
    """Portable default: compare actions from families available to this parent."""

    def resolve(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if type(context) is not CampaignPortfolioWaveContext:
            raise TypeError("context must be an exact CampaignPortfolioWaveContext")
        CampaignPortfolioWaveContext.__post_init__(context)
        families = tuple(
            sorted({option.family for option in context.variation.contract.options})
        )
        if not families:
            raise ValueError("eligible variation contract contains no action family")
        return families, ()


@dataclass(slots=True)
class ContextualOutcomeCampaignEnricher:
    """Expose bounded prior outcomes with explicit transfer distance.

    The returned object contains no card identity or model rationale.  Exact
    same-parent evidence is ranked first, then direct-lineage evidence.  When
    explicitly enabled, clearly labelled cross-lineage analogies rank last.
    The campaign runtime attaches it under its reserved contextual-history key.
    """

    ledger: PortfolioOutcomeFeedbackLedger
    max_actions: int = 24
    include_cross_lineage_analogies: bool = False
    relevance: CampaignOutcomeRelevanceResolver = AvailableFamilyOutcomeRelevance()

    def __post_init__(self) -> None:
        if type(self.ledger) is not PortfolioOutcomeFeedbackLedger:
            raise TypeError("ledger must be an exact PortfolioOutcomeFeedbackLedger")
        if type(self.max_actions) is not int or self.max_actions <= 0:
            raise ValueError("max_actions must be a positive exact integer")
        if type(self.include_cross_lineage_analogies) is not bool:
            raise TypeError("include_cross_lineage_analogies must be an exact boolean")
        if not isinstance(self.relevance, CampaignOutcomeRelevanceResolver):
            raise TypeError("relevance must implement CampaignOutcomeRelevanceResolver")

    @staticmethod
    def _lineage(context: CampaignPortfolioWaveContext) -> tuple[str, ...]:
        parent = context.parent
        values = {value.value for value in parent.parent_ids}
        if parent.common_ancestor_id is not None:
            values.add(parent.common_ancestor_id.value)
        return tuple(sorted(values))

    def enrich(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> FrozenJsonObject:
        self.__post_init__()
        if type(context) is not CampaignPortfolioWaveContext:
            raise TypeError("context must be an exact CampaignPortfolioWaveContext")
        CampaignPortfolioWaveContext.__post_init__(context)
        families, changed_paths = self.relevance.resolve(context)
        query = ContextualOutcomeQuery(
            current_parent_candidate_id=context.parent.candidate_id.value,
            current_parent_configuration_sha256=(
                context.parent.occurrence.configuration_hash
            ),
            cutoff_wave_index_exclusive=context.stage_request.step.generation,
            lineage_candidate_ids=self._lineage(context),
            families=families,
            changed_paths=changed_paths,
            max_actions=self.max_actions,
            include_cross_lineage_analogies=(self.include_cross_lineage_analogies),
        )
        return self.ledger.contextual_history(query).to_prompt_record()


__all__ = [
    "AvailableFamilyOutcomeRelevance",
    "CampaignOutcomeRelevanceResolver",
    "ContextualOutcomeCampaignEnricher",
]
