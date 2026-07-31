"""Atomic learning-state publication for residual portfolio evolution.

One residual decision produces two coupled learning products:

* immediate, conserved action outcomes for the workload-blind broker; and
* proposal provenance, stage credit, and earned reproduction tickets.

Publishing those products into independent mutable ledgers can expose a
half-committed state after an exception or concurrent update.  This module
instead prepares both products against private ledger copies and commits them
with one compare-and-swap of a versioned aggregate state.  Proposal experts,
archive utilities, workloads, models, and providers remain outside this
boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_evolve.application.earned_lineage import (
    CandidateProposalProvenance,
    EarnedLineageLedger,
    ReproductionTicketIssuance,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionEvidenceLedger,
)
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioEvolutionResult,
)
from agent_evolve.application.residual_stage_credit import (
    ConservedResidualStageProjection,
    ResidualStageCreditProjector,
)
from agent_evolve.domain.patch import require_sha256


RESIDUAL_LEARNING_TRANSACTION_ID = "atomic_residual_learning_state"
RESIDUAL_LEARNING_TRANSACTION_VERSION = 1
RESIDUAL_LEARNING_TRANSACTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:atomic-residual-learning-state:v1;"
    b"prepare=private-exact-ledger-copies;"
    b"publication=single-versioned-state-compare-and-swap;"
    b"products=broker-outcomes-plus-provenance-credit-and-earned-tickets;"
    b"cutoff=authenticated-conserved-residual-stage-projection;"
    b"stale-preparations=fail-closed;"
    b"workload-model-provider-branches=false"
).hexdigest()

_STATE_DOMAIN = b"agent-evolve:residual-learning-state:v1\x00"
_PREPARATION_DOMAIN = b"agent-evolve:residual-learning-preparation:v1\x00"
_COMMIT_DOMAIN = b"agent-evolve:residual-learning-commit:v1\x00"


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


def _broker_record(ledger: MaterializedActionEvidenceLedger) -> dict[str, object]:
    if type(ledger) is not MaterializedActionEvidenceLedger:
        raise TypeError("broker evidence must be an exact ledger")
    return {
        "outcomes": [value.to_record() for value in ledger.outcomes],
        "delayed_credits": [
            value.to_record() for value in ledger.delayed_credits
        ],
        "resolved_returns": [
            value.to_record() for value in ledger.resolved_returns
        ],
    }


@dataclass(frozen=True, slots=True)
class ResidualLearningState:
    """One immutable-by-ownership broker and earned-lineage state revision."""

    broker_evidence: MaterializedActionEvidenceLedger = field(
        repr=False,
        compare=False,
    )
    earned_lineage: EarnedLineageLedger = field(
        repr=False,
        compare=False,
    )
    current_generation: int
    revision: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.broker_evidence) is not MaterializedActionEvidenceLedger:
            raise TypeError("broker_evidence must be an exact ledger")
        if type(self.earned_lineage) is not EarnedLineageLedger:
            raise TypeError("earned_lineage must be an exact ledger")
        self.earned_lineage.__post_init__()
        for name in ("current_generation", "revision"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        object.__setattr__(
            self,
            "state_sha256",
            _hash(_STATE_DOMAIN, self._unsigned_record()),
        )

    @classmethod
    def empty(cls) -> "ResidualLearningState":
        return cls(
            broker_evidence=MaterializedActionEvidenceLedger(),
            earned_lineage=EarnedLineageLedger(),
            current_generation=0,
            revision=0,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "revision": self.revision,
            "current_generation": self.current_generation,
            "broker_evidence": _broker_record(self.broker_evidence),
            "earned_lineage": self.earned_lineage.to_record(
                current_generation=self.current_generation
            ),
            "workload_model_provider_fields_present": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResidualLearningPublicationPreparation:
    """Fully validated successor state that has not yet been published."""

    base_state_sha256: str
    projection: ConservedResidualStageProjection
    ticket_issuance: ReproductionTicketIssuance
    next_state: ResidualLearningState = field(repr=False, compare=False)
    transaction_definition_sha256: str = (
        RESIDUAL_LEARNING_TRANSACTION_DEFINITION_SHA256
    )
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.base_state_sha256, "base_state_sha256")
        require_sha256(
            self.transaction_definition_sha256,
            "transaction_definition_sha256",
        )
        if type(self.projection) is not ConservedResidualStageProjection:
            raise TypeError("projection must be exact")
        self.projection.__post_init__()
        if type(self.ticket_issuance) is not ReproductionTicketIssuance:
            raise TypeError("ticket_issuance must be exact")
        self.ticket_issuance.__post_init__()
        if type(self.next_state) is not ResidualLearningState:
            raise TypeError("next_state must be exact")
        self.next_state.__post_init__()
        if self.next_state.state_sha256 == self.base_state_sha256:
            raise ValueError("learning publication must advance state")
        if (
            self.ticket_issuance.stage_credit_receipt_sha256
            != self.projection.stage_credit.receipt_sha256
        ):
            raise ValueError("ticket issuance differs from projected stage credit")
        if (
            self.ticket_issuance.generation
            != self.projection.stage_credit.generation
            or self.next_state.current_generation
            != self.projection.stage_credit.generation
        ):
            raise ValueError("learning publication mixes generations")
        object.__setattr__(
            self,
            "preparation_sha256",
            _hash(_PREPARATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "transaction": {
                "transaction_id": RESIDUAL_LEARNING_TRANSACTION_ID,
                "transaction_version": RESIDUAL_LEARNING_TRANSACTION_VERSION,
                "definition_sha256": self.transaction_definition_sha256,
            },
            "base_state_sha256": self.base_state_sha256,
            "projection_sha256": self.projection.projection_sha256,
            "ticket_issuance_sha256": self.ticket_issuance.issuance_sha256,
            "next_state_sha256": self.next_state.state_sha256,
            "publication_performed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "projection": self.projection.to_record(),
            "ticket_issuance": self.ticket_issuance.to_record(),
            "preparation_sha256": self.preparation_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResidualLearningPublicationCommit:
    """Receipt for one successful aggregate-state compare-and-swap."""

    preparation_sha256: str
    prior_state_sha256: str
    committed_state_sha256: str
    committed_revision: int
    commit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "preparation_sha256",
            "prior_state_sha256",
            "committed_state_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.prior_state_sha256 == self.committed_state_sha256:
            raise ValueError("commit must advance learning state")
        if type(self.committed_revision) is not int or self.committed_revision <= 0:
            raise ValueError("committed_revision must be positive")
        object.__setattr__(
            self,
            "commit_sha256",
            _hash(_COMMIT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "preparation_sha256": self.preparation_sha256,
            "prior_state_sha256": self.prior_state_sha256,
            "committed_state_sha256": self.committed_state_sha256,
            "committed_revision": self.committed_revision,
            "publication": "single_aggregate_state_compare_and_swap",
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "commit_sha256": self.commit_sha256}


@dataclass(slots=True)
class TransactionalResidualLearningStore:
    """Prepare complete stage learning state and publish it atomically."""

    projector: ResidualStageCreditProjector
    _state: ResidualLearningState = field(
        default_factory=ResidualLearningState.empty,
    )
    _prepared: dict[str, ResidualLearningPublicationPreparation] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if type(self.projector) is not ResidualStageCreditProjector:
            raise TypeError("projector must be an exact ResidualStageCreditProjector")
        self.projector.__post_init__()
        if type(self._state) is not ResidualLearningState:
            raise TypeError("_state must be an exact ResidualLearningState")
        self._state.__post_init__()

    @property
    def state(self) -> ResidualLearningState:
        """Return the current read-only-by-ownership state revision."""

        return self._state

    def register_prior_provenance(
        self,
        values: Iterable[CandidateProposalProvenance],
    ) -> ResidualLearningState:
        """Atomically register seed or imported provenance before a stage."""

        if self._prepared:
            raise RuntimeError("cannot register provenance with an open preparation")
        batch = tuple(values)
        if not batch:
            raise ValueError("prior provenance batch cannot be empty")
        next_lineage = copy.deepcopy(self._state.earned_lineage)
        next_lineage.register(batch)
        next_state = ResidualLearningState(
            broker_evidence=copy.deepcopy(self._state.broker_evidence),
            earned_lineage=next_lineage,
            current_generation=self._state.current_generation,
            revision=self._state.revision + 1,
        )
        self._state = next_state
        return next_state

    def prepare(
        self,
        *,
        pre_snapshot: ArchiveUtilitySnapshot,
        post_snapshot: ArchiveUtilitySnapshot,
        result: ResidualPortfolioEvolutionResult,
    ) -> ResidualLearningPublicationPreparation:
        """Project, preflight, and stage both learning ledgers without publishing."""

        if self._prepared:
            raise RuntimeError("only one residual learning preparation may be open")
        base = self._state
        projection = self.projector.project(
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            result=result,
        )
        generation = projection.stage_credit.generation
        if generation <= base.current_generation:
            raise ValueError("residual learning generations must increase strictly")

        next_broker = copy.deepcopy(base.broker_evidence)
        for outcome in projection.action_outcomes:
            next_broker.append_outcome(outcome)
        next_lineage = copy.deepcopy(base.earned_lineage)
        next_lineage.register(projection.candidate_provenance)
        issuance = next_lineage.observe(projection.stage_credit)
        next_state = ResidualLearningState(
            broker_evidence=next_broker,
            earned_lineage=next_lineage,
            current_generation=generation,
            revision=base.revision + 1,
        )
        preparation = ResidualLearningPublicationPreparation(
            base_state_sha256=base.state_sha256,
            projection=projection,
            ticket_issuance=issuance,
            next_state=next_state,
        )
        if preparation.preparation_sha256 in self._prepared:
            raise RuntimeError("learning preparation identity collision")
        self._prepared[preparation.preparation_sha256] = preparation
        return preparation

    def abort(self, preparation: ResidualLearningPublicationPreparation) -> None:
        preparation.__post_init__()
        known = self._prepared.pop(preparation.preparation_sha256, None)
        if known is not preparation:
            raise ValueError("preparation is absent, stale, or already closed")

    def commit(
        self,
        preparation: ResidualLearningPublicationPreparation,
    ) -> ResidualLearningPublicationCommit:
        """Publish one fully prepared successor state with a stale-state guard."""

        preparation.__post_init__()
        known = self._prepared.get(preparation.preparation_sha256)
        if known is not preparation:
            raise ValueError("preparation is absent, stale, or already closed")
        if self._state.state_sha256 != preparation.base_state_sha256:
            raise RuntimeError("learning state advanced after preparation")
        prior = self._state
        # This assignment is the only publication mutation.  Both underlying
        # ledgers were already validated on private copies in ``prepare``.
        self._state = preparation.next_state
        del self._prepared[preparation.preparation_sha256]
        return ResidualLearningPublicationCommit(
            preparation_sha256=preparation.preparation_sha256,
            prior_state_sha256=prior.state_sha256,
            committed_state_sha256=self._state.state_sha256,
            committed_revision=self._state.revision,
        )


__all__ = [
    "RESIDUAL_LEARNING_TRANSACTION_DEFINITION_SHA256",
    "RESIDUAL_LEARNING_TRANSACTION_ID",
    "RESIDUAL_LEARNING_TRANSACTION_VERSION",
    "ResidualLearningPublicationCommit",
    "ResidualLearningPublicationPreparation",
    "ResidualLearningState",
    "TransactionalResidualLearningStore",
]
