"""Generic transactional campaign stage for residual portfolio evolution.

The runtime composes four independent boundaries:

1. workload-owned proposal experts materialize candidate actions;
2. a workload/model/provider-blind broker selects expensive evaluations;
3. a workload-owned archive port prepares the resulting state transition; and
4. conserved broker and lineage learning publish through one aggregate state.

The archive port follows the same prepare/commit/abort rule as the learning
store: preparation may perform work, while commit is synchronous, no-I/O, and
must not fail for its own validated preparation.  This keeps the application
core generic without pretending that archive ownership belongs to the broker.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionExplorationPort,
    MaterializedActionReturnValuePort,
    MaterializedSlateFeasibilityPort,
    MaterializedSlateValuePort,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.prequential_residual_exploration import (
    PrequentialLowDiscrepancyResidualExploration,
)
from agent_evolve.application.residual_learning_transaction import (
    ResidualLearningPublicationCommit,
    ResidualLearningPublicationPreparation,
    TransactionalResidualLearningStore,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionAllocationPolicyPort,
    MaterializedActionProposalExpertPort,
    ResidualPortfolioDecisionRequest,
    ResidualPortfolioEvolution,
    ResidualPortfolioEvolutionResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


RESIDUAL_CAMPAIGN_RUNTIME_ID = "transactional_residual_portfolio_campaign"
RESIDUAL_CAMPAIGN_RUNTIME_VERSION = 3
RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:transactional-residual-portfolio-campaign:v3;"
    b"flow=expert-propose-optional-async-allocation-global-broker-selected-only-evaluate;"
    b"archive=injected-prepare-commit-abort-port;"
    b"learning=atomic-conserved-broker-and-earned-lineage-state;"
    b"commit-order=fully-prepare-archive-and-learning-then-no-io-publication;"
    b"broker-cutoff=prior-learning-state-only;"
    b"exploration=prequential-low-discrepancy-protected-actions;"
    b"workload-model-provider-branches=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_ARCHIVE_PREPARATION_DOMAIN = (
    b"agent-evolve:residual-archive-transition-preparation:v1\x00"
)
_ARCHIVE_COMMIT_DOMAIN = b"agent-evolve:residual-archive-transition-commit:v1\x00"
_STAGE_RECEIPT_DOMAIN = b"agent-evolve:residual-campaign-stage-receipt:v1\x00"


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


def _token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


@dataclass(frozen=True, slots=True)
class ResidualArchiveTransitionPreparation:
    """Workload-owned archive transition frozen before publication."""

    archive_id: str
    archive_version: int
    archive_definition_sha256: str
    residual_result_sha256: str
    pre_snapshot: ArchiveUtilitySnapshot
    post_snapshot: ArchiveUtilitySnapshot
    evidence: FrozenJsonObject
    preparation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.archive_id, name="archive_id")
        if type(self.archive_version) is not int or self.archive_version <= 0:
            raise ValueError("archive_version must be positive")
        require_sha256(
            self.archive_definition_sha256,
            "archive_definition_sha256",
        )
        require_sha256(self.residual_result_sha256, "residual_result_sha256")
        if type(self.pre_snapshot) is not ArchiveUtilitySnapshot:
            raise TypeError("pre_snapshot must be exact")
        if type(self.post_snapshot) is not ArchiveUtilitySnapshot:
            raise TypeError("post_snapshot must be exact")
        self.pre_snapshot.__post_init__()
        self.post_snapshot.__post_init__()
        if (
            self.pre_snapshot.utility_id,
            self.pre_snapshot.utility_version,
            self.pre_snapshot.definition_sha256,
            self.pre_snapshot.generation,
            self.pre_snapshot.benchmark_sha256,
        ) != (
            self.post_snapshot.utility_id,
            self.post_snapshot.utility_version,
            self.post_snapshot.definition_sha256,
            self.post_snapshot.generation,
            self.post_snapshot.benchmark_sha256,
        ):
            raise ValueError("archive transition snapshots are not comparable")
        if self.pre_snapshot.archive_sha256 == self.post_snapshot.archive_sha256:
            raise ValueError("archive transition must advance its state receipt")
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("archive evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "preparation_sha256",
            _hash(_ARCHIVE_PREPARATION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "archive": {
                "archive_id": self.archive_id,
                "archive_version": self.archive_version,
                "definition_sha256": self.archive_definition_sha256,
            },
            "residual_result_sha256": self.residual_result_sha256,
            "pre_snapshot_sha256": self.pre_snapshot.snapshot_sha256,
            "post_snapshot_sha256": self.post_snapshot.snapshot_sha256,
            "evidence_sha256": typed_json_sha256(self.evidence),
            "publication_performed": False,
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "pre_snapshot": self.pre_snapshot.to_record(),
            "post_snapshot": self.post_snapshot.to_record(),
            "preparation_sha256": self.preparation_sha256,
        }
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@dataclass(frozen=True, slots=True)
class ResidualArchiveTransitionCommit:
    """Typed receipt from the archive port's no-I/O publication."""

    preparation_sha256: str
    committed_archive_sha256: str
    evidence: FrozenJsonObject
    commit_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.preparation_sha256, "preparation_sha256")
        require_sha256(
            self.committed_archive_sha256,
            "committed_archive_sha256",
        )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("commit evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "commit_sha256",
            _hash(_ARCHIVE_COMMIT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "preparation_sha256": self.preparation_sha256,
            "committed_archive_sha256": self.committed_archive_sha256,
            "evidence_sha256": typed_json_sha256(self.evidence),
            "publication": "synchronous_no_io_prevalidated",
        }

    def to_record(self, *, include_evidence: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "commit_sha256": self.commit_sha256}
        if include_evidence:
            record["evidence"] = thaw_json(self.evidence)
        return record


@runtime_checkable
class ResidualArchiveTransitionPort(Protocol):
    """Prepare and publish a workload-owned archive transition."""

    archive_id: str
    archive_version: int
    definition_sha256: str

    async def prepare(
        self,
        result: ResidualPortfolioEvolutionResult,
    ) -> ResidualArchiveTransitionPreparation: ...

    def commit(
        self,
        preparation: ResidualArchiveTransitionPreparation,
    ) -> ResidualArchiveTransitionCommit: ...

    def abort(self, preparation: ResidualArchiveTransitionPreparation) -> None: ...


def _archive_identity(
    value: ResidualArchiveTransitionPort,
) -> tuple[str, int, str]:
    if not isinstance(value, ResidualArchiveTransitionPort):
        raise TypeError("archive must implement ResidualArchiveTransitionPort")
    identity = (
        getattr(value, "archive_id", None),
        getattr(value, "archive_version", None),
        getattr(value, "definition_sha256", None),
    )
    _token(identity[0], name="archive_id")
    if type(identity[1]) is not int or identity[1] <= 0:
        raise ValueError("archive_version must be positive")
    require_sha256(identity[2], "archive definition_sha256")
    return identity  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ResidualCampaignStageReceipt:
    """Complete trace join for one committed residual campaign decision."""

    result: ResidualPortfolioEvolutionResult
    archive_preparation: ResidualArchiveTransitionPreparation
    archive_commit: ResidualArchiveTransitionCommit
    learning_preparation: ResidualLearningPublicationPreparation
    learning_commit: ResidualLearningPublicationCommit
    runtime_definition_sha256: str = RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.result) is not ResidualPortfolioEvolutionResult:
            raise TypeError("result must be exact")
        self.result.__post_init__()
        if type(self.archive_preparation) is not ResidualArchiveTransitionPreparation:
            raise TypeError("archive_preparation must be exact")
        self.archive_preparation.__post_init__()
        if type(self.archive_commit) is not ResidualArchiveTransitionCommit:
            raise TypeError("archive_commit must be exact")
        self.archive_commit.__post_init__()
        if (
            type(self.learning_preparation)
            is not ResidualLearningPublicationPreparation
        ):
            raise TypeError("learning_preparation must be exact")
        self.learning_preparation.__post_init__()
        if type(self.learning_commit) is not ResidualLearningPublicationCommit:
            raise TypeError("learning_commit must be exact")
        self.learning_commit.__post_init__()
        require_sha256(
            self.runtime_definition_sha256,
            "runtime_definition_sha256",
        )
        if (
            self.archive_preparation.residual_result_sha256
            != self.result.result_sha256
            or self.learning_preparation.projection.residual_result_sha256
            != self.result.result_sha256
        ):
            raise ValueError("campaign stage products name different results")
        if (
            self.archive_commit.preparation_sha256
            != self.archive_preparation.preparation_sha256
            or self.archive_commit.committed_archive_sha256
            != self.archive_preparation.post_snapshot.archive_sha256
        ):
            raise ValueError("archive commit differs from its preparation")
        if (
            self.learning_commit.preparation_sha256
            != self.learning_preparation.preparation_sha256
            or self.learning_commit.committed_state_sha256
            != self.learning_preparation.next_state.state_sha256
        ):
            raise ValueError("learning commit differs from its preparation")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_STAGE_RECEIPT_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtime": {
                "runtime_id": RESIDUAL_CAMPAIGN_RUNTIME_ID,
                "runtime_version": RESIDUAL_CAMPAIGN_RUNTIME_VERSION,
                "definition_sha256": self.runtime_definition_sha256,
            },
            "result_sha256": self.result.result_sha256,
            "archive_preparation_sha256": (
                self.archive_preparation.preparation_sha256
            ),
            "archive_commit_sha256": self.archive_commit.commit_sha256,
            "learning_preparation_sha256": (
                self.learning_preparation.preparation_sha256
            ),
            "learning_commit_sha256": self.learning_commit.commit_sha256,
            "workload_model_provider_fields_present": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "result": self.result.to_record(
                include_allocation_evidence=True,
            ),
            "archive_transition": {
                "preparation": self.archive_preparation.to_record(),
                "commit": self.archive_commit.to_record(),
            },
            "learning_transaction": {
                "preparation": self.learning_preparation.to_record(),
                "commit": self.learning_commit.to_record(),
            },
            "receipt_sha256": self.receipt_sha256,
        }


async def commit_residual_campaign_stage(
    *,
    archive: ResidualArchiveTransitionPort,
    learning: TransactionalResidualLearningStore,
    result: ResidualPortfolioEvolutionResult,
) -> ResidualCampaignStageReceipt:
    """Atomically publish one already-evaluated generic residual result.

    This transaction boundary is deliberately independent of how the
    proposal/evaluation slate was produced.  Single-wave and sequential
    pilot-gated search therefore share exactly the same archive and learning
    publication semantics.
    """

    archive_identity = _archive_identity(archive)
    if type(learning) is not TransactionalResidualLearningStore:
        raise TypeError("learning must be an exact transactional store")
    learning.__post_init__()
    if type(result) is not ResidualPortfolioEvolutionResult:
        raise TypeError("result must be exact")
    result.__post_init__()

    archive_preparation = await archive.prepare(result)
    if type(archive_preparation) is not ResidualArchiveTransitionPreparation:
        raise TypeError("archive returned a foreign preparation")
    archive_preparation.__post_init__()
    if (
        archive_preparation.archive_id,
        archive_preparation.archive_version,
        archive_preparation.archive_definition_sha256,
    ) != archive_identity:
        raise ValueError("archive preparation has a foreign definition")
    if archive_preparation.residual_result_sha256 != result.result_sha256:
        raise ValueError("archive prepared a different residual result")

    learning_preparation: ResidualLearningPublicationPreparation | None = None
    archive_committed = False
    try:
        learning_preparation = learning.prepare(
            pre_snapshot=archive_preparation.pre_snapshot,
            post_snapshot=archive_preparation.post_snapshot,
            result=result,
        )
        archive_commit = archive.commit(archive_preparation)
        if type(archive_commit) is not ResidualArchiveTransitionCommit:
            raise TypeError("archive returned a foreign commit")
        archive_commit.__post_init__()
        archive_committed = True
        learning_commit = learning.commit(learning_preparation)
    except BaseException:
        if learning_preparation is not None and not archive_committed:
            learning.abort(learning_preparation)
        if not archive_committed:
            archive.abort(archive_preparation)
        raise
    return ResidualCampaignStageReceipt(
        result=result,
        archive_preparation=archive_preparation,
        archive_commit=archive_commit,
        learning_preparation=learning_preparation,
        learning_commit=learning_commit,
    )


@dataclass(frozen=True, slots=True)
class ResidualPortfolioCampaignStageRuntime:
    """Run and transactionally close one generic residual campaign stage."""

    experts: tuple[MaterializedActionProposalExpertPort, ...]
    archive: ResidualArchiveTransitionPort = field(repr=False, compare=False)
    learning: TransactionalResidualLearningStore = field(
        repr=False,
        compare=False,
    )
    slate_value: MaterializedSlateValuePort = field(repr=False, compare=False)
    slate_feasibility: MaterializedSlateFeasibilityPort = field(
        repr=False,
        compare=False,
    )
    allocation_policy: MaterializedActionAllocationPolicyPort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    return_value: MaterializedActionReturnValuePort | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    exploration_policy: MaterializedActionExplorationPort | None = field(
        default_factory=PrequentialLowDiscrepancyResidualExploration,
        repr=False,
        compare=False,
    )
    hierarchical_kappa: float = 4.0
    confidence_width: float = 1.0
    exact_combination_limit: int = 250_000
    beam_width: int = 512

    def __post_init__(self) -> None:
        if type(self.experts) is not tuple or not self.experts:
            raise ValueError("experts must be a non-empty exact tuple")
        _archive_identity(self.archive)
        if type(self.learning) is not TransactionalResidualLearningStore:
            raise TypeError("learning must be an exact transactional store")
        self.learning.__post_init__()
        if self.exploration_policy is not None and not isinstance(
            self.exploration_policy,
            MaterializedActionExplorationPort,
        ):
            raise TypeError(
                "exploration_policy must implement "
                "MaterializedActionExplorationPort"
            )
        if self.allocation_policy is not None and not isinstance(
            self.allocation_policy,
            MaterializedActionAllocationPolicyPort,
        ):
            raise TypeError(
                "allocation_policy must implement "
                "MaterializedActionAllocationPolicyPort"
            )
        for name in ("hierarchical_kappa", "confidence_width"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite exact float")
        for name in ("exact_combination_limit", "beam_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")

    async def run(
        self,
        request: ResidualPortfolioDecisionRequest,
    ) -> ResidualCampaignStageReceipt:
        self.__post_init__()
        runtime = ResidualPortfolioEvolution(
            experts=self.experts,
            broker=RegretBrokeredMaterializedActionPolicy(
                ledger=self.learning.state.broker_evidence,
                return_value=self.return_value,
                exploration_policy=self.exploration_policy,
                hierarchical_kappa=self.hierarchical_kappa,
                confidence_width=self.confidence_width,
                exact_combination_limit=self.exact_combination_limit,
                beam_width=self.beam_width,
            ),
            slate_value=self.slate_value,
            slate_feasibility=self.slate_feasibility,
            allocation_policy=self.allocation_policy,
        )
        result = await runtime.run(request)
        return await commit_residual_campaign_stage(
            archive=self.archive,
            learning=self.learning,
            result=result,
        )


__all__ = [
    "RESIDUAL_CAMPAIGN_RUNTIME_DEFINITION_SHA256",
    "RESIDUAL_CAMPAIGN_RUNTIME_ID",
    "RESIDUAL_CAMPAIGN_RUNTIME_VERSION",
    "ResidualArchiveTransitionCommit",
    "ResidualArchiveTransitionPort",
    "ResidualArchiveTransitionPreparation",
    "ResidualCampaignStageReceipt",
    "ResidualPortfolioCampaignStageRuntime",
    "commit_residual_campaign_stage",
]
