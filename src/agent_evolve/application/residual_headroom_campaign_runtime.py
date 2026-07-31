"""Transactional campaign boundary for residual-headroom learning.

The wrapped outcome-adaptive runtime publishes the authoritative archive and
ordinary learning transaction first.  Only after that transaction closes does
this module project conserved conditional credit and advance an injected
headroom ledger.  The ordering prevents an uncommitted stage from influencing
the next generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.outcome_adaptive_residual_campaign_runtime import (
    OutcomeAdaptiveResidualCampaignStageReceipt,
    OutcomeAdaptiveResidualPortfolioCampaignStageRuntime,
)
from agent_evolve.application.residual_headroom_ledger import (
    ConservedResidualHeadroomLedger,
    ConservedResidualHeadroomProjector,
    ResidualHeadroomAdaptiveMarketProjector,
    ResidualHeadroomLedgerState,
    ResidualHeadroomStageClosure,
)
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_ID = (
    "transactional_residual_headroom_campaign_runtime"
)
RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_VERSION = 1
RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:transactional-residual-headroom-campaign:v1;"
    b"delegate=outcome-adaptive-authoritative-archive-transaction;"
    b"ordering=archive-and-learning-commit-before-headroom-projection;"
    b"credit=conserved-conditional-set-gain;"
    b"next-generation-read=committed-prior-state-only;"
    b"counterfactual-paired-arm-credit=excluded;"
    b"workload-objective-model-provider-prompt-config-branches=false"
).hexdigest()

IN_MEMORY_RESIDUAL_HEADROOM_STORE_ID = (
    "in_memory_transactional_residual_headroom_store"
)
IN_MEMORY_RESIDUAL_HEADROOM_STORE_VERSION = 1
IN_MEMORY_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:in-memory-residual-headroom-store:v1;"
    b"compare-and-swap=expected-prior-state-sha256;"
    b"append=conserved-ledger-fold;"
    b"durable=false"
).hexdigest()

_ACK_DOMAIN = b"agent-evolve:residual-headroom-ledger-commit-ack:v1\x00"
_RECEIPT_DOMAIN = (
    b"agent-evolve:residual-headroom-campaign-stage-receipt:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ResidualHeadroomLedgerCommitAck:
    store_id: str
    store_version: int
    store_definition_sha256: str
    prior_state_sha256: str
    closure_sha256: str
    new_state_sha256: str
    durable: bool
    evidence: FrozenJsonObject
    ack_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.store_id) is not str or not self.store_id:
            raise ValueError("store_id must be a non-empty exact string")
        if type(self.store_version) is not int or self.store_version <= 0:
            raise ValueError("store_version must be positive")
        for name in (
            "store_definition_sha256",
            "prior_state_sha256",
            "closure_sha256",
            "new_state_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.prior_state_sha256 == self.new_state_sha256:
            raise ValueError("a ledger commit must advance its state")
        if type(self.durable) is not bool:
            raise TypeError("durable must be exact")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("commit evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "ack_sha256",
            _hash(_ACK_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "store": {
                "store_id": self.store_id,
                "store_version": self.store_version,
                "definition_sha256": self.store_definition_sha256,
            },
            "prior_state_sha256": self.prior_state_sha256,
            "closure_sha256": self.closure_sha256,
            "new_state_sha256": self.new_state_sha256,
            "durable": self.durable,
            "evidence_sha256": typed_json_sha256(self.evidence),
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "ack_sha256": self.ack_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class ResidualHeadroomLedgerCommitPort(Protocol):
    store_id: str
    store_version: int
    definition_sha256: str
    ledger: ConservedResidualHeadroomLedger
    state: ResidualHeadroomLedgerState

    async def commit(
        self,
        *,
        expected_prior_state_sha256: str,
        closure: ResidualHeadroomStageClosure,
    ) -> ResidualHeadroomLedgerCommitAck: ...


@dataclass(slots=True)
class InMemoryTransactionalResidualHeadroomStore:
    """Compare-and-swap store for tests and single-process development."""

    ledger: ConservedResidualHeadroomLedger
    state: ResidualHeadroomLedgerState
    store_id: str = IN_MEMORY_RESIDUAL_HEADROOM_STORE_ID
    store_version: int = IN_MEMORY_RESIDUAL_HEADROOM_STORE_VERSION
    definition_sha256: str = (
        IN_MEMORY_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256
    )
    commit_acks: tuple[ResidualHeadroomLedgerCommitAck, ...] = ()

    def __post_init__(self) -> None:
        if type(self.ledger) is not ConservedResidualHeadroomLedger:
            raise TypeError("ledger must be exact")
        self.ledger.__post_init__()
        if type(self.state) is not ResidualHeadroomLedgerState:
            raise TypeError("state must be exact")
        self.state.__post_init__()
        if (
            self.state.config_definition_sha256
            != self.ledger.config.definition_sha256
        ):
            raise ValueError("state belongs to another ledger configuration")
        require_sha256(self.definition_sha256, "definition_sha256")
        if type(self.commit_acks) is not tuple:
            raise TypeError("commit_acks must be an exact tuple")
        for value in self.commit_acks:
            if type(value) is not ResidualHeadroomLedgerCommitAck:
                raise TypeError("commit_acks must contain exact values")
            value.__post_init__()

    async def commit(
        self,
        *,
        expected_prior_state_sha256: str,
        closure: ResidualHeadroomStageClosure,
    ) -> ResidualHeadroomLedgerCommitAck:
        self.__post_init__()
        require_sha256(
            expected_prior_state_sha256,
            "expected_prior_state_sha256",
        )
        if expected_prior_state_sha256 != self.state.state_sha256:
            raise ValueError("headroom state compare-and-swap failed")
        new_state = self.ledger.append(self.state, closure)
        ack = ResidualHeadroomLedgerCommitAck(
            store_id=self.store_id,
            store_version=self.store_version,
            store_definition_sha256=self.definition_sha256,
            prior_state_sha256=self.state.state_sha256,
            closure_sha256=closure.closure_sha256,
            new_state_sha256=new_state.state_sha256,
            durable=False,
            evidence=freeze_json(
                {
                    "storage": "process_memory",
                    "append_only": True,
                }
            ),
        )
        self.state = new_state
        self.commit_acks = (*self.commit_acks, ack)
        return ack


@dataclass(frozen=True, slots=True)
class ResidualHeadroomCampaignStageReceipt:
    """Join an authoritative stage to its post-commit headroom transition."""

    delegate: OutcomeAdaptiveResidualCampaignStageReceipt
    prior_headroom_state_sha256: str
    headroom_closure: ResidualHeadroomStageClosure
    headroom_commit_ack: ResidualHeadroomLedgerCommitAck
    runtime_definition_sha256: str = (
        RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_DEFINITION_SHA256
    )
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.delegate)
            is not OutcomeAdaptiveResidualCampaignStageReceipt
        ):
            raise TypeError("delegate must be an exact campaign receipt")
        self.delegate.__post_init__()
        require_sha256(
            self.prior_headroom_state_sha256,
            "prior_headroom_state_sha256",
        )
        if type(self.headroom_closure) is not ResidualHeadroomStageClosure:
            raise TypeError("headroom_closure must be exact")
        self.headroom_closure.__post_init__()
        if (
            type(self.headroom_commit_ack)
            is not ResidualHeadroomLedgerCommitAck
        ):
            raise TypeError("headroom_commit_ack must be exact")
        self.headroom_commit_ack.__post_init__()
        require_sha256(
            self.runtime_definition_sha256,
            "runtime_definition_sha256",
        )
        adaptive = self.delegate.adaptive_result
        if (
            self.headroom_closure.residual_request_sha256
            != adaptive.result.request.request_sha256
            or self.headroom_commit_ack.prior_state_sha256
            != self.prior_headroom_state_sha256
            or self.headroom_commit_ack.closure_sha256
            != self.headroom_closure.closure_sha256
        ):
            raise ValueError(
                "headroom transition differs from its authoritative stage"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtime": {
                "runtime_id": RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_ID,
                "runtime_version": RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_VERSION,
                "definition_sha256": self.runtime_definition_sha256,
            },
            "delegate_receipt_sha256": self.delegate.receipt_sha256,
            "prior_headroom_state_sha256": (
                self.prior_headroom_state_sha256
            ),
            "headroom_closure_sha256": (
                self.headroom_closure.closure_sha256
            ),
            "headroom_commit_ack_sha256": (
                self.headroom_commit_ack.ack_sha256
            ),
            "authoritative_archive_committed_before_headroom": True,
            "paired_counterfactual_arm_credit_included": False,
            "workload_objective_model_provider_prompt_config_fields": False,
        }

    @property
    def adaptive_result(self):
        return self.delegate.adaptive_result

    @property
    def campaign_stage(self):
        return self.delegate.campaign_stage

    @property
    def result(self):
        return self.delegate.result

    @property
    def archive_preparation(self):
        return self.delegate.archive_preparation

    @property
    def archive_commit(self):
        return self.delegate.archive_commit

    @property
    def learning_preparation(self):
        return self.delegate.learning_preparation

    @property
    def learning_commit(self):
        return self.delegate.learning_commit

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "delegate": self.delegate.to_record(
                include_evidence=include_evidence
            ),
            "headroom_closure": self.headroom_closure.to_record(),
            "headroom_commit_ack": self.headroom_commit_ack.to_record(
                include_evidence=include_evidence
            ),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class TransactionalResidualHeadroomCampaignStageRuntime:
    """Decorate one adaptive stage with post-publication headroom learning."""

    delegate: OutcomeAdaptiveResidualPortfolioCampaignStageRuntime = field(
        repr=False,
        compare=False,
    )
    headroom_projector: ConservedResidualHeadroomProjector
    headroom_store: ResidualHeadroomLedgerCommitPort = field(
        repr=False,
        compare=False,
    )
    context_sha256: str
    require_durable_headroom_commit: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.delegate)
            is not OutcomeAdaptiveResidualPortfolioCampaignStageRuntime
        ):
            raise TypeError("delegate must be an exact adaptive runtime")
        self.delegate.__post_init__()
        if (
            type(self.headroom_projector)
            is not ConservedResidualHeadroomProjector
        ):
            raise TypeError("headroom_projector must be exact")
        if not isinstance(
            self.headroom_store,
            ResidualHeadroomLedgerCommitPort,
        ):
            raise TypeError("headroom_store must implement its port")
        self.headroom_store.state.__post_init__()
        require_sha256(self.context_sha256, "context_sha256")
        if type(self.require_durable_headroom_commit) is not bool:
            raise TypeError(
                "require_durable_headroom_commit must be exact"
            )
        market = self.delegate.market_projector
        if (
            type(market) is ResidualHeadroomAdaptiveMarketProjector
            and (
                market.ledger_state.state_sha256
                != self.headroom_store.state.state_sha256
                or market.context_sha256 != self.context_sha256
            )
        ):
            raise ValueError(
                "market projector is not bound to the committed prior state"
            )

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> ResidualHeadroomCampaignStageReceipt:
        self.__post_init__()
        prior_state_sha256 = self.headroom_store.state.state_sha256
        delegate_receipt = await self.delegate.run(request)
        adaptive = delegate_receipt.adaptive_result
        closure = self.headroom_projector.project(
            context_sha256=self.context_sha256,
            generation_index=request.decision_index,
            reference_gain_scale=(
                self.delegate.racing_policy.reference_gain_scale
            ),
            reference_gain_evidence_sha256=(
                self.delegate.racing_policy
                .reference_gain_evidence_sha256
            ),
            actions=adaptive.adaptive_actions,
            diagnostic_decision=adaptive.diagnostic_decision,
            continuation_decisions=adaptive.continuation_decisions,
            outcomes=adaptive.outcomes,
            set_outcomes=adaptive.set_outcomes,
        )
        ack = await self.headroom_store.commit(
            expected_prior_state_sha256=prior_state_sha256,
            closure=closure,
        )
        if (
            self.require_durable_headroom_commit
            and not ack.durable
        ):
            raise RuntimeError("headroom commit was not durable")
        return ResidualHeadroomCampaignStageReceipt(
            delegate=delegate_receipt,
            prior_headroom_state_sha256=prior_state_sha256,
            headroom_closure=closure,
            headroom_commit_ack=ack,
        )


__all__ = [
    "IN_MEMORY_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256",
    "IN_MEMORY_RESIDUAL_HEADROOM_STORE_ID",
    "IN_MEMORY_RESIDUAL_HEADROOM_STORE_VERSION",
    "InMemoryTransactionalResidualHeadroomStore",
    "RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_ID",
    "RESIDUAL_HEADROOM_CAMPAIGN_RUNTIME_VERSION",
    "ResidualHeadroomCampaignStageReceipt",
    "ResidualHeadroomLedgerCommitAck",
    "ResidualHeadroomLedgerCommitPort",
    "TransactionalResidualHeadroomCampaignStageRuntime",
]
