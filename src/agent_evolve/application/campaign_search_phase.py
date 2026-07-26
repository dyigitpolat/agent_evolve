"""Authenticated, workload-neutral phase context for campaign acquisition.

Selection policies need to know whether an evaluated slot can still create
downstream descendants.  Without that fact a fixed exploration quota wastes
terminal evaluations, while workload adapters are tempted to smuggle schedule
knowledge into prompts.  This module exposes only prior schedule geometry: the
current portfolio-generation ordinal and the number of later portfolio
generations.  It contains no objective values, workload vocabulary, model
identity, or outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)


CAMPAIGN_SEARCH_PHASE_CONTEXT_KEY = "campaign_search_phase"
CAMPAIGN_SEARCH_PHASE_POLICY_ID = "portfolio_horizon_phase"
CAMPAIGN_SEARCH_PHASE_POLICY_VERSION = 1
CAMPAIGN_SEARCH_PHASE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-horizon-phase:v1;"
    b"inputs=prepared-portfolio-generation-indices,current-generation;"
    b"outputs=ordinal,total,remaining;outcomes=false;"
    b"workload-model-provider-fields=false"
).hexdigest()
_CONTEXT_DOMAIN = b"agent-evolve:campaign-search-phase-context:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class CampaignSearchPhaseContext:
    """Prior-only schedule position attached to every portfolio request."""

    campaign_generation: int
    portfolio_generation_ordinal: int
    total_portfolio_generations: int
    remaining_portfolio_generations: int
    context_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_generation",
            "portfolio_generation_ordinal",
            "total_portfolio_generations",
            "remaining_portfolio_generations",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.campaign_generation <= 0:
            raise ValueError("campaign_generation must be positive")
        if self.total_portfolio_generations <= 0:
            raise ValueError("total_portfolio_generations must be positive")
        if not 1 <= self.portfolio_generation_ordinal <= (
            self.total_portfolio_generations
        ):
            raise ValueError("portfolio_generation_ordinal lies outside the schedule")
        expected_remaining = (
            self.total_portfolio_generations - self.portfolio_generation_ordinal
        )
        if self.remaining_portfolio_generations != expected_remaining:
            raise ValueError("remaining portfolio horizon differs from the ordinal")
        object.__setattr__(
            self,
            "context_sha256",
            hashlib.sha256(
                _CONTEXT_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def terminal(self) -> bool:
        return self.remaining_portfolio_generations == 0

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_policy": {
                "policy_id": CAMPAIGN_SEARCH_PHASE_POLICY_ID,
                "policy_version": CAMPAIGN_SEARCH_PHASE_POLICY_VERSION,
                "definition_sha256": (
                    CAMPAIGN_SEARCH_PHASE_POLICY_DEFINITION_SHA256
                ),
            },
            "campaign_generation": self.campaign_generation,
            "portfolio_generation_ordinal": self.portfolio_generation_ordinal,
            "total_portfolio_generations": self.total_portfolio_generations,
            "remaining_portfolio_generations": (
                self.remaining_portfolio_generations
            ),
            "current_or_future_outcomes_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "context_sha256": self.context_sha256}

    @classmethod
    def from_record(cls, record: object) -> "CampaignSearchPhaseContext":
        if type(record) is not dict or set(record) != {
            "schema_version",
            "phase_policy",
            "campaign_generation",
            "portfolio_generation_ordinal",
            "total_portfolio_generations",
            "remaining_portfolio_generations",
            "current_or_future_outcomes_consulted",
            "context_sha256",
        }:
            raise ValueError("campaign search phase record has an invalid field set")
        if record["schema_version"] != 1:
            raise ValueError("campaign search phase record has an unknown schema")
        if record["phase_policy"] != {
            "policy_id": CAMPAIGN_SEARCH_PHASE_POLICY_ID,
            "policy_version": CAMPAIGN_SEARCH_PHASE_POLICY_VERSION,
            "definition_sha256": CAMPAIGN_SEARCH_PHASE_POLICY_DEFINITION_SHA256,
        }:
            raise ValueError("campaign search phase policy identity differs")
        if record["current_or_future_outcomes_consulted"] is not False:
            raise ValueError("campaign search phase cannot consult outcomes")
        value = cls(
            campaign_generation=record["campaign_generation"],
            portfolio_generation_ordinal=record["portfolio_generation_ordinal"],
            total_portfolio_generations=record["total_portfolio_generations"],
            remaining_portfolio_generations=record[
                "remaining_portfolio_generations"
            ],
        )
        if record["context_sha256"] != value.context_sha256:
            raise ValueError("campaign search phase authentication failed")
        return value


def campaign_search_phase_context(
    *,
    campaign_generation: int,
    portfolio_generations: tuple[int, ...],
) -> CampaignSearchPhaseContext:
    """Build the unique phase record for a prepared campaign schedule."""

    if (
        type(portfolio_generations) is not tuple
        or not portfolio_generations
        or any(type(value) is not int or value <= 0 for value in portfolio_generations)
        or portfolio_generations != tuple(sorted(set(portfolio_generations)))
    ):
        raise ValueError("portfolio_generations must be positive and canonical")
    try:
        ordinal = portfolio_generations.index(campaign_generation) + 1
    except ValueError as error:
        raise ValueError(
            "campaign_generation is not a portfolio generation"
        ) from error
    return CampaignSearchPhaseContext(
        campaign_generation=campaign_generation,
        portfolio_generation_ordinal=ordinal,
        total_portfolio_generations=len(portfolio_generations),
        remaining_portfolio_generations=len(portfolio_generations) - ordinal,
    )


def attach_campaign_search_phase_context(
    context: FrozenJsonObject,
    phase: CampaignSearchPhaseContext,
) -> FrozenJsonObject:
    """Append the reserved phase subtree without changing existing fields."""

    if type(context) is not FrozenJsonObject or freeze_json(context) is not context:
        raise TypeError("context must be an exact frozen object")
    if type(phase) is not CampaignSearchPhaseContext:
        raise TypeError("phase must be exact")
    phase.__post_init__()
    values = thaw_json(context)
    if type(values) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("campaign context did not thaw to an object")
    if CAMPAIGN_SEARCH_PHASE_CONTEXT_KEY in values:
        raise ValueError("context already uses the reserved campaign phase key")
    values[CAMPAIGN_SEARCH_PHASE_CONTEXT_KEY] = phase.to_record()
    frozen = freeze_json(values)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("campaign phase context did not freeze to an object")
    return frozen


def resolve_campaign_search_phase_context(
    context: FrozenJsonObject,
) -> CampaignSearchPhaseContext:
    """Authenticate the phase subtree from one selection request."""

    if type(context) is not FrozenJsonObject or freeze_json(context) is not context:
        raise TypeError("context must be an exact frozen object")
    values = thaw_json(context)
    if type(values) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("campaign context did not thaw to an object")
    record = values.get(CAMPAIGN_SEARCH_PHASE_CONTEXT_KEY)
    if record is None:
        raise ValueError("campaign context omits the reserved search phase")
    return CampaignSearchPhaseContext.from_record(record)


__all__ = [
    "CAMPAIGN_SEARCH_PHASE_CONTEXT_KEY",
    "CAMPAIGN_SEARCH_PHASE_POLICY_DEFINITION_SHA256",
    "CAMPAIGN_SEARCH_PHASE_POLICY_ID",
    "CAMPAIGN_SEARCH_PHASE_POLICY_VERSION",
    "CampaignSearchPhaseContext",
    "attach_campaign_search_phase_context",
    "campaign_search_phase_context",
    "resolve_campaign_search_phase_context",
]
