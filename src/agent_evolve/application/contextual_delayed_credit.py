"""Workload-neutral delayed frontier, lineage, and terminal credit projection.

The service joins framework candidate lineage to previously sealed contextual
observations.  It never reads candidate configuration fields, objective names,
model prose, or provider metadata.  Recombination credit is selection-
conditioned evidence: a source receives a positive descendant verdict only
when at least one evaluated direct recombinant using it survives on the sealed
post-stage front.  Unselected sources remain unobserved for that channel rather
than being mislabeled as failures.  Every source candidate independently
receives post-stage frontier-survival credit.  Terminal persistence is
adjudicated separately after successful campaign completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.application.contextual_search_controller import (
    ContextualSearchDelayedCredit,
    ContextualSearchObservation,
)
from agent_evolve.application.portfolio_evolution import PortfolioMemberDisposition
from agent_evolve.application.portfolio_recombination import (
    PortfolioRecombinationWaveResult,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256


_POST_RECOMBINATION_BATCH_DOMAIN = (
    b"agent-evolve:contextual-post-recombination-credit-batch:v1\x00"
)
_PERSISTENCE_BATCH_DOMAIN = (
    b"agent-evolve:contextual-terminal-persistence-credit-batch:v1\x00"
)
CONTEXTUAL_DELAYED_CREDIT_POLICY_ID = "sealed_multi_horizon_contextual_credit"
CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION = 2
CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:sealed-multi-horizon-contextual-credit:v2;"
    b"lineage=framework-candidate-ids;stage-survival=all-source-candidates-on-"
    b"sealed-post-stage-front;descendant=selected-direct-recombinant-survives-"
    b"sealed-post-stage-front;unselected-descendant=unobserved;"
    b"persistence=successful-campaign-terminal-front-membership;"
    b"availability=strictly-after-source-wave;"
    b"workload-objective-model-provider-fields=false"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _canonical_candidate_ids(
    values: tuple[CandidateId, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[CandidateId, ...]:
    if type(values) is not tuple or any(
        type(value) is not CandidateId for value in values
    ):
        raise TypeError(f"{name} must contain exact CandidateId values")
    for value in values:
        CandidateId.__post_init__(value)
    if not allow_empty and not values:
        raise ValueError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")
    return values


@dataclass(frozen=True, slots=True)
class ContextualPostRecombinationCreditBatch:
    campaign_scope_sha256: str
    source_wave_index: int
    available_at_wave_index: int
    recombination_receipt_sha256s: tuple[str, ...]
    selected_source_candidate_ids: tuple[CandidateId, ...]
    stage_surviving_source_candidate_ids: tuple[CandidateId, ...]
    useful_descendant_candidate_ids: tuple[CandidateId, ...]
    post_stage_front_candidate_ids: tuple[CandidateId, ...]
    credits: tuple[ContextualSearchDelayedCredit, ...]
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.source_wave_index) is not int or self.source_wave_index <= 0:
            raise ValueError("source_wave_index must be positive")
        if (
            type(self.available_at_wave_index) is not int
            or self.available_at_wave_index <= self.source_wave_index
        ):
            raise ValueError("descendant credit must become available later")
        if (
            type(self.recombination_receipt_sha256s) is not tuple
            or not self.recombination_receipt_sha256s
            or self.recombination_receipt_sha256s
            != tuple(sorted(set(self.recombination_receipt_sha256s)))
        ):
            raise ValueError("recombination receipts must be non-empty and canonical")
        for value in self.recombination_receipt_sha256s:
            require_sha256(value, "recombination_receipt_sha256")
        _canonical_candidate_ids(
            self.selected_source_candidate_ids,
            name="selected_source_candidate_ids",
        )
        _canonical_candidate_ids(
            self.stage_surviving_source_candidate_ids,
            name="stage_surviving_source_candidate_ids",
            allow_empty=True,
        )
        _canonical_candidate_ids(
            self.useful_descendant_candidate_ids,
            name="useful_descendant_candidate_ids",
            allow_empty=True,
        )
        _canonical_candidate_ids(
            self.post_stage_front_candidate_ids,
            name="post_stage_front_candidate_ids",
        )
        front = set(self.post_stage_front_candidate_ids)
        if not set(self.stage_surviving_source_candidate_ids).issubset(front):
            raise ValueError("stage survivors escape the post-stage front")
        if not set(self.useful_descendant_candidate_ids).issubset(front):
            raise ValueError("useful descendants escape the post-stage front")
        if (
            type(self.credits) is not tuple
            or not self.credits
            or any(
                type(value) is not ContextualSearchDelayedCredit
                for value in self.credits
            )
        ):
            raise ValueError("credits must contain exact delayed-credit values")
        for value in self.credits:
            value.__post_init__()
            if (
                value.campaign_scope_sha256 != self.campaign_scope_sha256
                or value.available_at_wave_index != self.available_at_wave_index
                or value.final_front_persisted is not None
                or value.stage_front_persisted is None
            ):
                raise ValueError("post-recombination credit differs from its batch")
        if tuple(value.credit_sha256 for value in self.credits) != tuple(
            sorted({value.credit_sha256 for value in self.credits})
        ):
            raise ValueError("post-recombination credits must be unique and canonical")
        object.__setattr__(
            self,
            "batch_sha256",
            hashlib.sha256(
                _POST_RECOMBINATION_BATCH_DOMAIN
                + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "source_wave_index": self.source_wave_index,
            "available_at_wave_index": self.available_at_wave_index,
            "recombination_receipt_sha256s": list(self.recombination_receipt_sha256s),
            "selected_source_candidate_ids": [
                value.value for value in self.selected_source_candidate_ids
            ],
            "stage_surviving_source_candidate_ids": [
                value.value for value in self.stage_surviving_source_candidate_ids
            ],
            "useful_descendant_candidate_ids": [
                value.value for value in self.useful_descendant_candidate_ids
            ],
            "post_stage_front_candidate_ids": [
                value.value for value in self.post_stage_front_candidate_ids
            ],
            "credit_sha256s": [value.credit_sha256 for value in self.credits],
            "policy": {
                "policy_id": CONTEXTUAL_DELAYED_CREDIT_POLICY_ID,
                "policy_version": CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION,
                "definition_sha256": CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "credits": [value.to_record() for value in self.credits],
            "batch_sha256": self.batch_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextualTerminalPersistenceCreditBatch:
    campaign_scope_sha256: str
    available_at_wave_index: int
    finalization_request_sha256: str
    terminal_front_candidate_ids: tuple[CandidateId, ...]
    credits: tuple[ContextualSearchDelayedCredit, ...]
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        require_sha256(
            self.finalization_request_sha256,
            "finalization_request_sha256",
        )
        if (
            type(self.available_at_wave_index) is not int
            or self.available_at_wave_index <= 1
        ):
            raise ValueError("terminal credit availability must follow search")
        _canonical_candidate_ids(
            self.terminal_front_candidate_ids,
            name="terminal_front_candidate_ids",
        )
        if (
            type(self.credits) is not tuple
            or not self.credits
            or any(
                type(value) is not ContextualSearchDelayedCredit
                for value in self.credits
            )
        ):
            raise ValueError("credits must contain exact delayed-credit values")
        for value in self.credits:
            value.__post_init__()
            if (
                value.campaign_scope_sha256 != self.campaign_scope_sha256
                or value.available_at_wave_index != self.available_at_wave_index
                or value.stage_front_persisted is not None
                or value.final_front_persisted is None
                or value.useful_descendant_observed is not None
            ):
                raise ValueError("terminal persistence credit differs from its batch")
        if tuple(value.credit_sha256 for value in self.credits) != tuple(
            sorted({value.credit_sha256 for value in self.credits})
        ):
            raise ValueError("terminal credits must be unique and canonical")
        object.__setattr__(
            self,
            "batch_sha256",
            hashlib.sha256(
                _PERSISTENCE_BATCH_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "available_at_wave_index": self.available_at_wave_index,
            "finalization_request_sha256": self.finalization_request_sha256,
            "terminal_front_candidate_ids": [
                value.value for value in self.terminal_front_candidate_ids
            ],
            "credit_sha256s": [value.credit_sha256 for value in self.credits],
            "policy": {
                "policy_id": CONTEXTUAL_DELAYED_CREDIT_POLICY_ID,
                "policy_version": CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION,
                "definition_sha256": CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "credits": [value.to_record() for value in self.credits],
            "batch_sha256": self.batch_sha256,
        }


def observe_contextual_post_recombination_credit(
    *,
    campaign_scope_sha256: str,
    source_wave_index: int,
    observations: tuple[ContextualSearchObservation, ...],
    results: tuple[PortfolioRecombinationWaveResult, ...],
    post_stage_front_candidate_ids: tuple[CandidateId, ...],
) -> ContextualPostRecombinationCreditBatch:
    """Adjudicate source survival and selected-source descendant yield."""

    require_sha256(campaign_scope_sha256, "campaign_scope_sha256")
    if type(source_wave_index) is not int or source_wave_index <= 0:
        raise ValueError("source_wave_index must be positive")
    if (
        type(observations) is not tuple
        or not observations
        or any(type(value) is not ContextualSearchObservation for value in observations)
    ):
        raise ValueError("observations must contain exact contextual values")
    observation_by_candidate: dict[CandidateId, ContextualSearchObservation] = {}
    for value in observations:
        value.__post_init__()
        if (
            value.campaign_scope_sha256 != campaign_scope_sha256
            or value.wave_index != source_wave_index
        ):
            raise ValueError("observation differs from the source wave")
        if value.candidate_id is None:
            raise ValueError("delayed credit requires candidate-bound observations")
        if value.candidate_id in observation_by_candidate:
            raise ValueError("contextual source wave repeats a candidate")
        observation_by_candidate[value.candidate_id] = value
    if (
        type(results) is not tuple
        or not results
        or any(type(value) is not PortfolioRecombinationWaveResult for value in results)
    ):
        raise ValueError("results must contain exact recombination values")
    front = _canonical_candidate_ids(
        post_stage_front_candidate_ids,
        name="post_stage_front_candidate_ids",
    )
    front_set = set(front)
    stage_survivors = set(observation_by_candidate) & front_set
    selected_sources: set[CandidateId] = set()
    useful_descendants: set[CandidateId] = set()
    useful_by_source: dict[CandidateId, bool] = {}
    receipts: list[str] = []
    for result in results:
        result.__post_init__()
        receipts.append(result.receipt.receipt_sha256)
        for member in result.receipt.members:
            selected_sources.update(member.pair_ids)
            useful = (
                member.disposition is PortfolioMemberDisposition.SCORED
                and member.target_candidate_id in front_set
            )
            if useful:
                useful_descendants.add(member.target_candidate_id)
            for source_id in member.pair_ids:
                useful_by_source[source_id] = (
                    useful_by_source.get(source_id, False) or useful
                )
    missing = selected_sources - set(observation_by_candidate)
    if missing:
        raise ValueError("recombination cites a source without contextual observation")
    credits = tuple(
        sorted(
            (
                ContextualSearchDelayedCredit(
                    campaign_scope_sha256=campaign_scope_sha256,
                    source_observation_sha256=(
                        observation_by_candidate[candidate_id].observation_sha256
                    ),
                    available_at_wave_index=source_wave_index + 1,
                    stage_front_persisted=candidate_id in stage_survivors,
                    useful_descendant_observed=(
                        useful_by_source[candidate_id]
                        if candidate_id in selected_sources
                        else None
                    ),
                )
                for candidate_id in sorted(observation_by_candidate)
            ),
            key=lambda value: value.credit_sha256,
        )
    )
    return ContextualPostRecombinationCreditBatch(
        campaign_scope_sha256=campaign_scope_sha256,
        source_wave_index=source_wave_index,
        available_at_wave_index=source_wave_index + 1,
        recombination_receipt_sha256s=tuple(sorted(receipts)),
        selected_source_candidate_ids=tuple(sorted(selected_sources)),
        stage_surviving_source_candidate_ids=tuple(sorted(stage_survivors)),
        useful_descendant_candidate_ids=tuple(sorted(useful_descendants)),
        post_stage_front_candidate_ids=front,
        credits=credits,
    )


def observe_contextual_terminal_persistence(
    *,
    campaign_scope_sha256: str,
    available_at_wave_index: int,
    finalization_request_sha256: str,
    observations: tuple[ContextualSearchObservation, ...],
    terminal_front_candidate_ids: tuple[CandidateId, ...],
) -> ContextualTerminalPersistenceCreditBatch:
    """Adjudicate candidate persistence only after successful completion."""

    require_sha256(campaign_scope_sha256, "campaign_scope_sha256")
    require_sha256(finalization_request_sha256, "finalization_request_sha256")
    if (
        type(observations) is not tuple
        or not observations
        or any(type(value) is not ContextualSearchObservation for value in observations)
    ):
        raise ValueError("observations must contain exact contextual values")
    terminal = _canonical_candidate_ids(
        terminal_front_candidate_ids,
        name="terminal_front_candidate_ids",
    )
    terminal_set = set(terminal)
    for value in observations:
        value.__post_init__()
        if value.campaign_scope_sha256 != campaign_scope_sha256:
            raise ValueError("terminal credit crosses campaign scopes")
        if value.candidate_id is None:
            raise ValueError("terminal credit requires candidate-bound observations")
        if available_at_wave_index <= value.wave_index:
            raise ValueError("terminal credit predates a source observation")
    credits = tuple(
        sorted(
            (
                ContextualSearchDelayedCredit(
                    campaign_scope_sha256=campaign_scope_sha256,
                    source_observation_sha256=value.observation_sha256,
                    available_at_wave_index=available_at_wave_index,
                    final_front_persisted=value.candidate_id in terminal_set,
                )
                for value in observations
            ),
            key=lambda value: value.credit_sha256,
        )
    )
    return ContextualTerminalPersistenceCreditBatch(
        campaign_scope_sha256=campaign_scope_sha256,
        available_at_wave_index=available_at_wave_index,
        finalization_request_sha256=finalization_request_sha256,
        terminal_front_candidate_ids=terminal,
        credits=credits,
    )


__all__ = [
    "CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256",
    "CONTEXTUAL_DELAYED_CREDIT_POLICY_ID",
    "CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION",
    "ContextualPostRecombinationCreditBatch",
    "ContextualTerminalPersistenceCreditBatch",
    "observe_contextual_post_recombination_credit",
    "observe_contextual_terminal_persistence",
]
