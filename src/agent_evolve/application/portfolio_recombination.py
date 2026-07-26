"""Replay-safe exploit/coverage recombination over one ranked portfolio wave.

This application service contains no benchmark semantics and makes no provider
calls.  It accounts for every ranked child, excludes candidate-attributable
infeasibilities with authenticated no-resampling evidence, enumerates every
pair of remaining scored children, retains only exact disjoint
ancestor-relative unions, applies the existing exploit/coverage pair policy,
and evaluates the selected engine-materialized unions concurrently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Literal

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MaterializedInvocation,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.portfolio_evolution import (
    MaterializedPortfolioEngine,
    PortfolioCandidateFailureEvidence,
    PortfolioMemberDisposition,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
    require_sha256,
)
from agent_evolve.domain.typed_json import typed_json_equal, typed_json_sha256
from agent_evolve.policies.selection.disjoint_pairs import (
    DisjointBranchFacts,
    DisjointPairSelectionDecision,
    DisjointParentPairPolicy,
    ReplayVerifiedDisjointPair,
)
from agent_evolve.policies.selection.frozen_archive_pairs import (
    ArchiveAwareDisjointPairSelectionDecision,
    ArchiveAwareDisjointParentPairPolicy,
    FrozenArchiveBranchUtility,
    FrozenArchiveSourcePairUtility,
    FrozenArchiveSourceUtilityContext,
    FrozenArchiveSourceUtilityReceipt,
    ObservedSourceBranch,
)
from agent_evolve.policies.selection.task_keyed_palette import PathFamilyExposure
from agent_evolve.policies.variation.disjoint_recombination import (
    POLICY_ID as DISJOINT_PATCH_POLICY_ID,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    POLICY_VERSION as DISJOINT_PATCH_POLICY_VERSION,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchMaterialization,
    DisjointPatchRecombinationError,
    DisjointPatchRecombiner,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import SourceAttribution
from agent_evolve.ports.id_factory import IdFactory


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_WAVE_DOMAIN = b"agent-evolve:portfolio-recombination-wave:v1\x00"
_MEMBER_DOMAIN = b"agent-evolve:portfolio-recombination-member:v1\x00"
_SOURCE_EXCLUSION_DOMAIN = (
    b"agent-evolve:portfolio-recombination-source-exclusion:v1\x00"
)
_NO_PAIR_DOMAIN = b"agent-evolve:portfolio-recombination-no-pair:v1\x00"


def _canonical_json(record: dict[str, object]) -> bytes:
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash_record(domain: bytes, record: dict[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _require_token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def _canonical_exposures(
    values: tuple[PathFamilyExposure, ...],
) -> tuple[PathFamilyExposure, ...]:
    if type(values) is not tuple or any(
        type(value) is not PathFamilyExposure for value in values
    ):
        raise TypeError("path_family_exposures must contain exact values")
    for value in values:
        value.revalidate()
    keys = tuple((canonical_path_bytes(value.path), value.family) for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("path/family exposure cells must be unique")
    expected = tuple(
        sorted(
            values, key=lambda value: (canonical_path_bytes(value.path), value.family)
        )
    )
    if values != expected:
        raise ValueError("path_family_exposures must use canonical order")
    return values


def frozen_archive_source_utility_context(
    snapshot: ArchiveUtilitySnapshot,
) -> FrozenArchiveSourceUtilityContext:
    """Project the generic campaign snapshot into pair-policy authority."""

    if type(snapshot) is not ArchiveUtilitySnapshot:
        raise TypeError("snapshot must be an exact ArchiveUtilitySnapshot")
    ArchiveUtilitySnapshot.__post_init__(snapshot)
    return FrozenArchiveSourceUtilityContext(
        utility_id=snapshot.utility_id,
        utility_version=snapshot.utility_version,
        utility_definition_sha256=snapshot.definition_sha256,
        benchmark_sha256=snapshot.benchmark_sha256,
        archive_cutoff_sha256=snapshot.archive_sha256,
        archive_snapshot_sha256=snapshot.snapshot_sha256,
        snapshot_generation=snapshot.generation,
    )


def portfolio_recombination_observed_sources(
    source_result: PortfolioVariationWaveResult,
) -> tuple[ObservedSourceBranch, ...]:
    """Project scored source branches into utility evidence.

    Candidate-attributable infeasibilities remain part of the ranked
    intention-to-treat wave, but they have no decision-objective vector and
    therefore cannot receive archive utility or become recombination parents.
    Their separate authenticated exclusion receipts are emitted by the
    recombination wave.
    """

    if type(source_result) is not PortfolioVariationWaveResult:
        raise TypeError("source_result must be an exact PortfolioVariationWaveResult")
    PortfolioVariationWaveResult.__post_init__(source_result)
    return tuple(
        ObservedSourceBranch(
            source_rank=member.materialization.rank,
            candidate_id=member.materialization.candidate_id,
            candidate_configuration_sha256=(
                member.materialization.child_configuration_sha256
            ),
            source_outcome_sha256=member.outcome_sha256,
        )
        for member in source_result.receipt.members
        if member.disposition is PortfolioMemberDisposition.SCORED
    )


def bind_portfolio_recombination_source_utilities(
    *,
    snapshot: ArchiveUtilitySnapshot,
    source_wave: PortfolioVariationWaveRequest,
    source_result: PortfolioVariationWaveResult,
    marginal_utilities: Mapping[CandidateId, float],
    exact_pair_utilities: Mapping[tuple[CandidateId, CandidateId], float],
) -> FrozenArchiveSourceUtilityReceipt:
    """Commit injected source and exact pair utilities without benchmark semantics.

    The caller-owned utility port computes the numbers.  This boundary only
    authenticates their archive cutoff, complete source universe, and canonical
    pair universe; it never names or interprets workload objectives.
    """

    if type(source_wave) is not PortfolioVariationWaveRequest:
        raise TypeError("source_wave must be an exact PortfolioVariationWaveRequest")
    PortfolioVariationWaveRequest.__post_init__(source_wave)
    if type(source_result) is not PortfolioVariationWaveResult:
        raise TypeError("source_result must be an exact PortfolioVariationWaveResult")
    PortfolioVariationWaveResult.__post_init__(source_result)
    receipt = source_result.receipt
    selection = source_wave.selection_request
    if (
        receipt.selection_call_id != selection.call_id
        or receipt.request_sha256 != selection.request_sha256
        or receipt.parent_candidate_id != source_wave.parent.candidate_id
        or receipt.parent_configuration_sha256
        != source_wave.parent.occurrence.configuration_hash
        or receipt.generation != source_wave.generation
    ):
        raise ValueError("source result differs from its exact wave request")
    context = frozen_archive_source_utility_context(snapshot)
    if context.snapshot_generation != source_wave.generation:
        raise ValueError("archive utility snapshot is stale for the source wave")
    sources = portfolio_recombination_observed_sources(source_result)
    source_ids = tuple(value.candidate_id for value in sources)
    if not isinstance(marginal_utilities, Mapping):
        raise TypeError("marginal_utilities must be a mapping")
    marginal_snapshot = dict(marginal_utilities)
    if set(marginal_snapshot) != set(source_ids) or any(
        type(value) is not CandidateId for value in marginal_snapshot
    ):
        raise ValueError("marginal utilities differ from the exact source universe")
    if not isinstance(exact_pair_utilities, Mapping):
        raise TypeError("exact_pair_utilities must be a mapping")
    pair_snapshot = dict(exact_pair_utilities)
    expected_pairs = tuple(combinations(sorted(source_ids), 2))
    if set(pair_snapshot) != set(expected_pairs) or any(
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(value) is not CandidateId for value in pair)
        for pair in pair_snapshot
    ):
        raise ValueError("exact pair utilities differ from the complete source pairs")
    contract = selection.finite_variation_contract
    return FrozenArchiveSourceUtilityReceipt(
        context=context,
        source_wave_receipt_sha256=receipt.receipt_sha256,
        source_request_sha256=selection.request_sha256,
        source_decision_sha256=receipt.decision_sha256,
        source_contract_sha256=contract.identity_sha256,
        source_generation=source_wave.generation,
        branches=tuple(
            FrozenArchiveBranchUtility(
                source=source,
                marginal_utility=marginal_snapshot[source.candidate_id],
            )
            for source in sources
        ),
        pair_utilities=tuple(
            FrozenArchiveSourcePairUtility(
                pair_ids=pair_ids,
                exact_joint_utility=pair_snapshot[pair_ids],
            )
            for pair_ids in expected_pairs
        ),
    )


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationWaveRequest:
    """Exact source wave, ancestor, generation, and optional exposure state."""

    source_wave: PortfolioVariationWaveRequest
    source_result: PortfolioVariationWaveResult
    ancestor: EvolutionCandidate
    generation: int
    label_prefix: str
    phase: str = "portfolio_recombination"
    path_family_exposures: tuple[PathFamilyExposure, ...] = ()
    source_archive_snapshot: ArchiveUtilitySnapshot | None = None
    source_utilities: FrozenArchiveSourceUtilityReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.source_wave) is not PortfolioVariationWaveRequest:
            raise TypeError("source_wave must be an exact request")
        PortfolioVariationWaveRequest.__post_init__(self.source_wave)
        if type(self.source_result) is not PortfolioVariationWaveResult:
            raise TypeError("source_result must be an exact result")
        PortfolioVariationWaveResult.__post_init__(self.source_result)
        if type(self.ancestor) is not EvolutionCandidate:
            raise TypeError("ancestor must be an exact EvolutionCandidate")
        EvolutionCandidate.__post_init__(self.ancestor)
        if self.ancestor != self.source_wave.parent:
            raise ValueError("ancestor differs from the source portfolio parent")
        receipt = self.source_result.receipt
        selection = self.source_wave.selection_request
        if (
            receipt.selection_call_id != selection.call_id
            or receipt.request_sha256 != selection.request_sha256
            or receipt.parent_candidate_id != self.ancestor.candidate_id
            or receipt.parent_configuration_sha256
            != self.ancestor.occurrence.configuration_hash
            or receipt.generation != self.source_wave.generation
        ):
            raise ValueError("source result differs from its exact wave request")
        contract = selection.finite_variation_contract
        for member, candidate in zip(
            receipt.members,
            self.source_result.candidates,
            strict=True,
        ):
            materialization = member.materialization
            option = contract.resolve(materialization.option_id)
            if (
                materialization.option_identity_sha256 != option.identity_sha256
                or materialization.child_configuration_sha256
                != option.child_configuration_sha256
                or candidate.candidate_id != materialization.candidate_id
                or not typed_json_equal(
                    candidate.configuration, option.child_configuration
                )
            ):
                raise ValueError(
                    "source branch differs from its finite option evidence"
                )
        if type(self.generation) is not int or self.generation <= max(
            candidate.generation for candidate in self.source_result.candidates
        ):
            raise ValueError("recombination generation must follow every source child")
        _require_token(self.label_prefix, "label_prefix")
        _require_token(self.phase, "phase")
        _canonical_exposures(self.path_family_exposures)
        if (self.source_archive_snapshot is None) != (self.source_utilities is None):
            raise ValueError(
                "source_archive_snapshot and source_utilities must be supplied together"
            )
        if self.source_archive_snapshot is not None:
            if type(self.source_archive_snapshot) is not ArchiveUtilitySnapshot:
                raise TypeError(
                    "source_archive_snapshot must be exact ArchiveUtilitySnapshot or None"
                )
            ArchiveUtilitySnapshot.__post_init__(self.source_archive_snapshot)
            if type(self.source_utilities) is not FrozenArchiveSourceUtilityReceipt:
                raise TypeError(
                    "source_utilities must be exact FrozenArchiveSourceUtilityReceipt"
                )
            expected_context = frozen_archive_source_utility_context(
                self.source_archive_snapshot
            )
            self.source_utilities.require_exact_context(
                context=expected_context,
                source_wave_receipt_sha256=receipt.receipt_sha256,
                source_request_sha256=selection.request_sha256,
                source_decision_sha256=receipt.decision_sha256,
                source_contract_sha256=contract.identity_sha256,
                source_generation=self.source_wave.generation,
                source_branches=portfolio_recombination_observed_sources(
                    self.source_result
                ),
            )


class PortfolioRecombinationSourceExclusionReason(str, Enum):
    """Closed reason a ranked ITT member cannot be a recombination parent."""

    CANDIDATE_INFEASIBLE = "candidate_infeasible"


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationSourceExclusionReceipt:
    """Authenticated non-resampling exclusion of one ranked source member."""

    rank: int
    candidate_id: CandidateId
    candidate_configuration_sha256: str
    option_id: str
    option_identity_sha256: str
    family: str
    source_outcome_sha256: str
    candidate_failure: PortfolioCandidateFailureEvidence
    reason: PortfolioRecombinationSourceExclusionReason = (
        PortfolioRecombinationSourceExclusionReason.CANDIDATE_INFEASIBLE
    )

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be exact")
        CandidateId.__post_init__(self.candidate_id)
        for name in (
            "candidate_configuration_sha256",
            "option_identity_sha256",
            "source_outcome_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        _require_token(self.family, "family")
        if type(self.candidate_failure) is not PortfolioCandidateFailureEvidence:
            raise TypeError("candidate_failure must be exact")
        PortfolioCandidateFailureEvidence.__post_init__(self.candidate_failure)
        if type(self.reason) is not PortfolioRecombinationSourceExclusionReason:
            raise TypeError("reason must be exact")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "rank": self.rank,
            "candidate_id": self.candidate_id.value,
            "candidate_configuration_sha256": self.candidate_configuration_sha256,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "source_outcome_sha256": self.source_outcome_sha256,
            "reason": self.reason.value,
            "candidate_failure": self.candidate_failure.to_record(),
            "resampled": False,
        }

    @property
    def exclusion_sha256(self) -> str:
        return _hash_record(_SOURCE_EXCLUSION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "exclusion_sha256": self.exclusion_sha256,
        }


class PortfolioRecombinationNoPairReason(str, Enum):
    """Closed terminal meanings for an empty recombination evaluation wave."""

    INSUFFICIENT_SCORED_SOURCES = "insufficient_scored_sources"
    NO_REPLAY_SAFE_DISJOINT_PAIR = "no_replay_safe_disjoint_pair"


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationNoPairReceipt:
    """Typed proof that no recombination child could be selected."""

    reason: PortfolioRecombinationNoPairReason
    scored_source_count: int
    excluded_source_count: int
    enumerated_pair_count: int
    replay_safe_pair_count: int

    def __post_init__(self) -> None:
        if type(self.reason) is not PortfolioRecombinationNoPairReason:
            raise TypeError("reason must be exact")
        for name in (
            "scored_source_count",
            "excluded_source_count",
            "enumerated_pair_count",
            "replay_safe_pair_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact non-negative integer")
        expected_pairs = self.scored_source_count * (
            self.scored_source_count - 1
        ) // 2
        if self.enumerated_pair_count != expected_pairs:
            raise ValueError("enumerated_pair_count does not close the source universe")
        if self.replay_safe_pair_count > self.enumerated_pair_count:
            raise ValueError("replay-safe count exceeds enumerated pair count")
        if self.replay_safe_pair_count != 0:
            raise ValueError("a no-pair receipt cannot hide a replay-safe pair")
        if self.reason is PortfolioRecombinationNoPairReason.INSUFFICIENT_SCORED_SOURCES:
            if self.scored_source_count >= 2:
                raise ValueError("insufficient-source reason requires fewer than two")
        elif self.scored_source_count < 2 or self.enumerated_pair_count == 0:
            raise ValueError("no-safe-pair reason requires an enumerated pair universe")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "reason": self.reason.value,
            "scored_source_count": self.scored_source_count,
            "excluded_source_count": self.excluded_source_count,
            "enumerated_pair_count": self.enumerated_pair_count,
            "replay_safe_pair_count": self.replay_safe_pair_count,
            "evaluation_wave_dispatched": False,
        }

    @property
    def no_pair_sha256(self) -> str:
        return _hash_record(_NO_PAIR_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "no_pair_sha256": self.no_pair_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationBranchBinding:
    """Rank, option family, reward, paths, and exposure for one source branch."""

    rank: int
    candidate_id: CandidateId
    candidate_configuration_sha256: str
    option_id: str
    option_identity_sha256: str
    family: str
    role: str
    reward: float
    changed_paths: tuple[str, ...]
    path_family_exposure: int
    source_outcome_sha256: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        if type(self.candidate_id) is not CandidateId:
            raise TypeError("candidate_id must be exact")
        CandidateId.__post_init__(self.candidate_id)
        for name in (
            "candidate_configuration_sha256",
            "option_identity_sha256",
            "source_outcome_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        _require_token(self.family, "family")
        _require_token(self.role, "role")
        if self.role != f"rank_{self.rank:04d}":
            raise ValueError("branch role must exactly encode its source rank")
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if (
            type(self.changed_paths) is not tuple
            or not self.changed_paths
            or any(
                type(path) is not str or not path.startswith("$.")
                for path in self.changed_paths
            )
            or self.changed_paths != tuple(sorted(set(self.changed_paths)))
        ):
            raise ValueError("changed_paths must be non-empty, unique, and canonical")
        if type(self.path_family_exposure) is not int or not (
            0 <= self.path_family_exposure <= (1 << 63) - 1
        ):
            raise ValueError("path_family_exposure must be an exact non-negative int63")

    @property
    def facts(self) -> DisjointBranchFacts:
        return DisjointBranchFacts(
            candidate_id=self.candidate_id,
            reward=self.reward,
            role=self.role,
            family=self.family,
            path_family_exposure=self.path_family_exposure,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id.value,
            "candidate_configuration_sha256": self.candidate_configuration_sha256,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "role": self.role,
            "reward_hex": self.reward.hex(),
            "changed_paths": list(self.changed_paths),
            "path_family_exposure": self.path_family_exposure,
            "source_outcome_sha256": self.source_outcome_sha256,
        }


@dataclass(frozen=True, slots=True)
class PortfolioPairAttemptReceipt:
    """One member of the complete n-choose-2 mechanical enumeration ledger."""

    left_candidate_id: CandidateId
    right_candidate_id: CandidateId
    target_candidate_id: CandidateId
    replay_safe: bool
    target_configuration_sha256: str | None = None
    materialization_receipt_sha256: str | None = None
    union_patch_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "left_candidate_id",
            "right_candidate_id",
            "target_candidate_id",
        ):
            value = getattr(self, name)
            if type(value) is not CandidateId:
                raise TypeError(f"{name} must be exact")
            CandidateId.__post_init__(value)
        if self.left_candidate_id >= self.right_candidate_id:
            raise ValueError("pair branch IDs must use canonical order")
        if self.target_candidate_id in {
            self.left_candidate_id,
            self.right_candidate_id,
        }:
            raise ValueError("pair target ID must differ from its branches")
        if type(self.replay_safe) is not bool:
            raise TypeError("replay_safe must be bool")
        optional = (
            self.target_configuration_sha256,
            self.materialization_receipt_sha256,
            self.union_patch_sha256,
        )
        if self.replay_safe != all(value is not None for value in optional):
            raise ValueError("safe status must agree with materialization evidence")
        for name in (
            "target_configuration_sha256",
            "materialization_receipt_sha256",
            "union_patch_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)

    @property
    def pair_ids(self) -> tuple[CandidateId, CandidateId]:
        return self.left_candidate_id, self.right_candidate_id

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "left_candidate_id": self.left_candidate_id.value,
            "right_candidate_id": self.right_candidate_id.value,
            "target_candidate_id": self.target_candidate_id.value,
            "replay_safe": self.replay_safe,
            "target_configuration_sha256": self.target_configuration_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "union_patch_sha256": self.union_patch_sha256,
        }


PortfolioRecombinationRole = Literal["exploit", "coverage"]
PortfolioRecombinationPairDecision = (
    DisjointPairSelectionDecision | ArchiveAwareDisjointPairSelectionDecision
)


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationMemberReceipt:
    """Closed pair-selection, materialization, and terminal-outcome join.

    A replay-safe materialization can still be rejected by a workload
    evaluator.  Candidate-attributable infeasibility is therefore a complete
    member outcome, not a partial recombination wave or an orchestration
    failure.  The disposition mirrors direct portfolio variation so the core
    applies one workload-independent terminal-outcome contract to both paths.
    """

    selection_role: PortfolioRecombinationRole
    pair_ids: tuple[CandidateId, CandidateId]
    source_ranks: tuple[int, int]
    source_option_ids: tuple[str, str]
    source_families: tuple[str, str]
    target_candidate_id: CandidateId
    target_configuration_sha256: str
    materialization_receipt_sha256: str
    union_patch_sha256: str
    parent_patch_sha256s: tuple[str, str]
    source_attribution: tuple[SourceAttribution, ...]
    operator_invocation_id: OperatorInvocationId
    reward_definition_sha256: str
    reward: float
    parent_relations: tuple[OutcomeRelation, ...]
    detailed_evaluation_sha256: str | None
    dominates_any_parent: bool
    better_than_any_parent: bool
    disposition: PortfolioMemberDisposition = PortfolioMemberDisposition.SCORED
    candidate_failure: PortfolioCandidateFailureEvidence | None = None

    def __post_init__(self) -> None:
        if self.selection_role not in {"exploit", "coverage"}:
            raise ValueError("selection_role must be exploit or coverage")
        if (
            type(self.pair_ids) is not tuple
            or len(self.pair_ids) != 2
            or any(type(value) is not CandidateId for value in self.pair_ids)
        ):
            raise TypeError("pair_ids must contain two exact CandidateId values")
        if self.pair_ids[0] >= self.pair_ids[1]:
            raise ValueError("pair_ids must use canonical order")
        if (
            type(self.source_ranks) is not tuple
            or len(self.source_ranks) != 2
            or any(type(value) is not int or value <= 0 for value in self.source_ranks)
        ):
            raise ValueError("source_ranks must contain two positive exact integers")
        for name in ("source_option_ids", "source_families"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) != 2
                or any(type(value) is not str or not value for value in values)
            ):
                raise ValueError(f"{name} must contain two non-empty strings")
        for family in self.source_families:
            _require_token(family, "source family")
        if type(self.target_candidate_id) is not CandidateId:
            raise TypeError("target_candidate_id must be exact")
        CandidateId.__post_init__(self.target_candidate_id)
        if self.target_candidate_id in self.pair_ids:
            raise ValueError("target candidate must differ from both parents")
        for name in (
            "target_configuration_sha256",
            "materialization_receipt_sha256",
            "union_patch_sha256",
            "reward_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.parent_patch_sha256s) is not tuple
            or len(self.parent_patch_sha256s) != 2
        ):
            raise TypeError("parent_patch_sha256s must contain two hashes")
        for value in self.parent_patch_sha256s:
            require_sha256(value, "parent patch SHA-256")
        if type(self.source_attribution) is not tuple or any(
            type(value) is not SourceAttribution for value in self.source_attribution
        ):
            raise TypeError("source_attribution must contain exact values")
        for value in self.source_attribution:
            SourceAttribution.__post_init__(value)
        if type(self.operator_invocation_id) is not OperatorInvocationId:
            raise TypeError("operator_invocation_id must be exact")
        OperatorInvocationId.__post_init__(self.operator_invocation_id)
        if type(self.reward) is not float or not math.isfinite(self.reward):
            raise TypeError("reward must be a finite canonical float")
        if type(self.parent_relations) is not tuple or any(
            type(value) is not OutcomeRelation for value in self.parent_relations
        ):
            raise TypeError("parent_relations must contain exact values")
        if self.detailed_evaluation_sha256 is not None:
            require_sha256(
                self.detailed_evaluation_sha256,
                "detailed_evaluation_sha256",
            )
        if (
            type(self.dominates_any_parent) is not bool
            or type(self.better_than_any_parent) is not bool
        ):
            raise TypeError("outcome comparison projections must be bool")
        if type(self.disposition) is not PortfolioMemberDisposition:
            raise TypeError("disposition must be a PortfolioMemberDisposition")
        failure = self.candidate_failure
        if self.disposition is PortfolioMemberDisposition.SCORED:
            if len(self.parent_relations) != 2:
                raise ValueError("scored recombination must compare to both parents")
            if failure is not None:
                raise ValueError(
                    "scored recombination cannot carry candidate failure evidence"
                )
        else:
            if type(failure) is not PortfolioCandidateFailureEvidence:
                raise TypeError(
                    "candidate-infeasible recombination requires exact failure evidence"
                )
            PortfolioCandidateFailureEvidence.__post_init__(failure)
            if self.detailed_evaluation_sha256 != failure.detailed_evaluation_sha256:
                raise ValueError(
                    "recombination failure identifies another detailed evaluation"
                )
            if self.parent_relations:
                raise ValueError(
                    "candidate-infeasible recombination cannot publish parent relations"
                )
            if self.dominates_any_parent or self.better_than_any_parent:
                raise ValueError(
                    "candidate-infeasible recombination cannot publish improvement flags"
                )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "selection_role": self.selection_role,
            "pair_ids": [value.value for value in self.pair_ids],
            "source_ranks": list(self.source_ranks),
            "source_option_ids": list(self.source_option_ids),
            "source_families": list(self.source_families),
            "target_candidate_id": self.target_candidate_id.value,
            "target_configuration_sha256": self.target_configuration_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "union_patch_sha256": self.union_patch_sha256,
            "parent_patch_sha256s": list(self.parent_patch_sha256s),
            "source_attribution": [
                {"path": value.path, "source": value.source}
                for value in self.source_attribution
            ],
            "operator_invocation_id": self.operator_invocation_id.value,
            "reward_definition_sha256": self.reward_definition_sha256,
            "disposition": self.disposition.value,
            "candidate_valid": (
                self.disposition is PortfolioMemberDisposition.SCORED
            ),
            "reward_hex": self.reward.hex(),
            "parent_relations": [value.value for value in self.parent_relations],
            "detailed_evaluation_sha256": self.detailed_evaluation_sha256,
            "candidate_failure": (
                None
                if self.candidate_failure is None
                else self.candidate_failure.to_record()
            ),
            "dominates_any_parent": self.dominates_any_parent,
            "better_than_any_parent": self.better_than_any_parent,
        }

    @property
    def outcome_sha256(self) -> str:
        return _hash_record(_MEMBER_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "outcome_sha256": self.outcome_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationWaveReceipt:
    """Complete pair universe, deterministic decision, and selected outcomes."""

    source_wave_receipt_sha256: str
    source_request_sha256: str
    source_decision_sha256: str
    source_contract_sha256: str
    ancestor_candidate_id: CandidateId
    ancestor_configuration_sha256: str
    generation: int
    branches: tuple[PortfolioRecombinationBranchBinding, ...]
    source_exclusions: tuple[PortfolioRecombinationSourceExclusionReceipt, ...]
    path_family_exposures: tuple[PathFamilyExposure, ...]
    pair_attempts: tuple[PortfolioPairAttemptReceipt, ...]
    pair_decision: PortfolioRecombinationPairDecision
    members: tuple[PortfolioRecombinationMemberReceipt, ...]
    selection_limit: int = 2
    no_pair: PortfolioRecombinationNoPairReceipt | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_wave_receipt_sha256",
            "source_request_sha256",
            "source_decision_sha256",
            "source_contract_sha256",
            "ancestor_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.ancestor_candidate_id) is not CandidateId:
            raise TypeError("ancestor_candidate_id must be exact")
        CandidateId.__post_init__(self.ancestor_candidate_id)
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.branches) is not tuple or any(
            type(value) is not PortfolioRecombinationBranchBinding
            for value in self.branches
        ):
            raise TypeError("branches must contain exact bindings")
        for value in self.branches:
            PortfolioRecombinationBranchBinding.__post_init__(value)
        branch_ranks = tuple(value.rank for value in self.branches)
        if branch_ranks != tuple(sorted(set(branch_ranks))):
            raise ValueError("branch bindings must remain in canonical ranked order")
        branch_ids = tuple(value.candidate_id for value in self.branches)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch candidate IDs must be unique")
        if type(self.source_exclusions) is not tuple or any(
            type(value) is not PortfolioRecombinationSourceExclusionReceipt
            for value in self.source_exclusions
        ):
            raise TypeError("source_exclusions must contain exact receipts")
        for value in self.source_exclusions:
            PortfolioRecombinationSourceExclusionReceipt.__post_init__(value)
        exclusion_ranks = tuple(value.rank for value in self.source_exclusions)
        if exclusion_ranks != tuple(sorted(set(exclusion_ranks))):
            raise ValueError("source exclusions must use canonical ranked order")
        source_ranks = tuple(sorted((*branch_ranks, *exclusion_ranks)))
        if source_ranks != tuple(range(1, len(source_ranks) + 1)):
            raise ValueError(
                "scored branches and exclusions must exactly cover ranked ITT members"
            )
        exclusion_ids = tuple(value.candidate_id for value in self.source_exclusions)
        if len(set(exclusion_ids)) != len(exclusion_ids) or set(branch_ids).intersection(
            exclusion_ids
        ):
            raise ValueError("source candidate IDs must be unique across dispositions")
        _canonical_exposures(self.path_family_exposures)
        exposure_by_cell = {
            (_path_text(value.path), value.family): value.count
            for value in self.path_family_exposures
        }
        for branch in self.branches:
            expected_exposure = sum(
                exposure_by_cell.get((path, branch.family), 0)
                for path in branch.changed_paths
            )
            if branch.path_family_exposure != expected_exposure:
                raise ValueError(
                    "branch exposure differs from the exact path/family snapshot"
                )
        if type(self.pair_attempts) is not tuple or any(
            type(value) is not PortfolioPairAttemptReceipt
            for value in self.pair_attempts
        ):
            raise TypeError("pair_attempts must contain exact receipts")
        for value in self.pair_attempts:
            PortfolioPairAttemptReceipt.__post_init__(value)
        expected_pair_ids = tuple(combinations(sorted(branch_ids), 2))
        if tuple(value.pair_ids for value in self.pair_attempts) != expected_pair_ids:
            raise ValueError("pair_attempts must enumerate the complete pair universe")
        target_ids = tuple(value.target_candidate_id for value in self.pair_attempts)
        if (
            len(set(target_ids)) != len(target_ids)
            or self.ancestor_candidate_id in set(target_ids)
            or set(target_ids) & set(branch_ids)
        ):
            raise ValueError("pair attempt target IDs collide with wave occurrences")
        if type(self.pair_decision) not in {
            DisjointPairSelectionDecision,
            ArchiveAwareDisjointPairSelectionDecision,
        }:
            raise TypeError("pair_decision must be an exact supported decision")
        self.pair_decision.revalidate()
        branch_by_id = {value.candidate_id: value for value in self.branches}
        safe_by_ids = {
            value.pair_ids: value for value in self.pair_attempts if value.replay_safe
        }
        rows_by_ids = {
            value.pair_ids: value for value in self.pair_decision.eligible_rows
        }
        if set(safe_by_ids) != set(rows_by_ids):
            raise ValueError("safe pair attempts differ from pair-policy eligibility")
        for pair_ids, row in rows_by_ids.items():
            attempt = safe_by_ids[pair_ids]
            if (
                row.pair.left != branch_by_id[pair_ids[0]].facts
                or row.pair.right != branch_by_id[pair_ids[1]].facts
                or row.pair.target_configuration_sha256
                != attempt.target_configuration_sha256
                or row.pair.materialization_receipt_sha256
                != attempt.materialization_receipt_sha256
            ):
                raise ValueError("eligible pair facts differ from enumeration evidence")
        if type(self.pair_decision) is ArchiveAwareDisjointPairSelectionDecision:
            source_utilities = self.pair_decision.source_utilities
            if source_utilities.source_generation >= self.generation:
                raise ValueError(
                    "archive-aware source utility must precede recombination generation"
                )
            source_utilities.require_exact_context(
                context=source_utilities.context,
                source_wave_receipt_sha256=self.source_wave_receipt_sha256,
                source_request_sha256=self.source_request_sha256,
                source_decision_sha256=self.source_decision_sha256,
                source_contract_sha256=self.source_contract_sha256,
                source_generation=source_utilities.source_generation,
                source_branches=tuple(
                    ObservedSourceBranch(
                        source_rank=value.rank,
                        candidate_id=value.candidate_id,
                        candidate_configuration_sha256=(
                            value.candidate_configuration_sha256
                        ),
                        source_outcome_sha256=value.source_outcome_sha256,
                    )
                    for value in self.branches
                ),
            )
        no_pair = self.no_pair
        if rows_by_ids:
            if no_pair is not None:
                raise ValueError("a replay-safe pair cannot carry no-pair evidence")
        else:
            if type(no_pair) is not PortfolioRecombinationNoPairReceipt:
                raise TypeError("an empty safe-pair universe requires no-pair evidence")
            PortfolioRecombinationNoPairReceipt.__post_init__(no_pair)
            expected_reason = (
                PortfolioRecombinationNoPairReason.INSUFFICIENT_SCORED_SOURCES
                if len(self.branches) < 2
                else PortfolioRecombinationNoPairReason.NO_REPLAY_SAFE_DISJOINT_PAIR
            )
            if (
                no_pair.reason is not expected_reason
                or no_pair.scored_source_count != len(self.branches)
                or no_pair.excluded_source_count != len(self.source_exclusions)
                or no_pair.enumerated_pair_count != len(self.pair_attempts)
                or no_pair.replay_safe_pair_count != 0
            ):
                raise ValueError("no-pair evidence differs from the exact pair universe")
        if type(self.members) is not tuple or any(
            type(value) is not PortfolioRecombinationMemberReceipt
            for value in self.members
        ):
            raise TypeError("members must contain exact receipts")
        if type(self.selection_limit) is not int or self.selection_limit not in {1, 2}:
            raise ValueError("selection_limit must be one or two")
        for value in self.members:
            PortfolioRecombinationMemberReceipt.__post_init__(value)
        expected_roles: list[tuple[str, tuple[CandidateId, CandidateId]]] = []
        if self.pair_decision.exploit_pair_ids is not None:
            expected_roles.append(("exploit", self.pair_decision.exploit_pair_ids))
        if self.pair_decision.coverage_pair_ids is not None:
            expected_roles.append(("coverage", self.pair_decision.coverage_pair_ids))
        expected_roles = expected_roles[: self.selection_limit]
        if tuple(
            (value.selection_role, value.pair_ids) for value in self.members
        ) != tuple(expected_roles):
            raise ValueError("member roles differ from the exact pair decision")
        for member in self.members:
            attempt = safe_by_ids[member.pair_ids]
            left, right = (branch_by_id[value] for value in member.pair_ids)
            if (
                member.source_ranks != (left.rank, right.rank)
                or member.source_option_ids != (left.option_id, right.option_id)
                or member.source_families != (left.family, right.family)
                or member.target_candidate_id != attempt.target_candidate_id
                or member.target_configuration_sha256
                != attempt.target_configuration_sha256
                or member.materialization_receipt_sha256
                != attempt.materialization_receipt_sha256
                or member.union_patch_sha256 != attempt.union_patch_sha256
            ):
                raise ValueError("selected member differs from branch/pair evidence")
        if len({value.target_candidate_id for value in self.members}) != len(
            self.members
        ) or len({value.operator_invocation_id for value in self.members}) != len(
            self.members
        ):
            raise ValueError("selected recombination members collide")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        record: dict[str, object] = {
            "schema_version": 1,
            "source_wave_receipt_sha256": self.source_wave_receipt_sha256,
            "source_request_sha256": self.source_request_sha256,
            "source_decision_sha256": self.source_decision_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "ancestor_candidate_id": self.ancestor_candidate_id.value,
            "ancestor_configuration_sha256": self.ancestor_configuration_sha256,
            "generation": self.generation,
            "branches": [value.to_record() for value in self.branches],
            "path_family_exposures": [
                value.to_trace_record() for value in self.path_family_exposures
            ],
            "pair_universe_size": len(self.pair_attempts),
            "pair_attempts": [value.to_record() for value in self.pair_attempts],
            "pair_decision": self.pair_decision.to_trace_record(),
            "selected_member_count": len(self.members),
            "concurrent_materialized_evaluation_wave": bool(self.members),
            "members": [value.to_record() for value in self.members],
        }
        if self.source_exclusions or self.no_pair is not None:
            record.update(
                {
                    "schema_version": 2,
                    "source_exclusions": [
                        value.to_record() for value in self.source_exclusions
                    ],
                    "ranked_source_count": len(self.branches)
                    + len(self.source_exclusions),
                    "scored_source_count": len(self.branches),
                    "excluded_source_count": len(self.source_exclusions),
                    "no_pair": (
                        None if self.no_pair is None else self.no_pair.to_record()
                    ),
                }
            )
        infeasible_members = tuple(
            value
            for value in self.members
            if value.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
        )
        if infeasible_members:
            record.update(
                {
                    "schema_version": 3,
                    "scored_member_count": len(self.members)
                    - len(infeasible_members),
                    "candidate_infeasible_member_count": len(infeasible_members),
                    "candidate_infeasibility_recourse": (
                        "retain_selected_itt_reject_from_archive_no_resampling"
                    ),
                }
            )
        if self.selection_limit != 2:
            record.update(
                {
                    "schema_version": 4,
                    "selection_limit": self.selection_limit,
                }
            )
        return record

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_WAVE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioRecombinationWaveResult:
    """Planner-facing evidence plus exact evaluated recombination outcomes."""

    receipt: PortfolioRecombinationWaveReceipt
    outcomes: tuple[InvocationOutcome, ...]

    def __post_init__(self) -> None:
        if type(self.receipt) is not PortfolioRecombinationWaveReceipt:
            raise TypeError("receipt must be an exact recombination receipt")
        PortfolioRecombinationWaveReceipt.__post_init__(self.receipt)
        if type(self.outcomes) is not tuple or any(
            type(value) is not InvocationOutcome for value in self.outcomes
        ):
            raise TypeError("outcomes must contain exact InvocationOutcome values")
        if len(self.outcomes) != len(self.receipt.members):
            raise ValueError("outcomes differ from receipt member count")
        for member, outcome in zip(self.receipt.members, self.outcomes, strict=True):
            InvocationOutcome.__post_init__(outcome)
            candidate = outcome.candidate
            prepared = outcome.prepared
            if (
                outcome.failure_stage is not None
                or candidate is None
                or not candidate.operator_compliant
                or not candidate.evidence_compliant
            ):
                raise ValueError("result contains a failed or partial outcome")
            detailed = candidate.detailed_evaluation
            detailed_sha256 = None if detailed is None else detailed.evidence_sha256
            disposition = PortfolioMemberDisposition.SCORED
            candidate_failure: PortfolioCandidateFailureEvidence | None = None
            if not candidate.valid:
                if detailed is None or detailed.failure is None:
                    raise ValueError(
                        "candidate infeasibility requires detailed evaluator evidence"
                    )
                candidate_failure = (
                    PortfolioCandidateFailureEvidence.from_failure_record(
                        detailed.failure,
                        detailed_evaluation_sha256=detailed.evidence_sha256,
                    )
                )
                disposition = PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
            plan = prepared.plan
            if (
                prepared.operator_invocation_id != member.operator_invocation_id
                or prepared.proposal_authority is not ProposalAuthority.ENGINE
                or prepared.call_id is not None
                or prepared.candidate_id != member.target_candidate_id
                or prepared.materialized_candidate_id != member.target_candidate_id
                or prepared.materialization_policy_id != DISJOINT_PATCH_POLICY_ID
                or prepared.materialization_policy_version
                != DISJOINT_PATCH_POLICY_VERSION
                or prepared.materialization_receipt_hash
                != member.materialization_receipt_sha256
                or prepared.variation_case.reward_definition_hash
                != member.reward_definition_sha256
                or plan.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION
                or tuple(value.candidate_id for value in plan.parents)
                != member.pair_ids
                or plan.common_ancestor is None
                or plan.common_ancestor.candidate_id
                != self.receipt.ancestor_candidate_id
                or plan.generation != self.receipt.generation
                or candidate.candidate_id != member.target_candidate_id
                or candidate.occurrence.configuration_hash
                != member.target_configuration_sha256
                or candidate.generation != self.receipt.generation
                or candidate.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION
                or candidate.parent_ids != member.pair_ids
                or candidate.common_ancestor_id != self.receipt.ancestor_candidate_id
                or candidate.parent_patch_hashes != member.parent_patch_sha256s
                or candidate.preservation_verified is not True
                or candidate.call_telemetry is not None
                or candidate.source_attribution != member.source_attribution
                or outcome.reward != member.reward
                or outcome.parent_relations != member.parent_relations
                or outcome.dominates_any_parent != member.dominates_any_parent
                or outcome.better_than_any_parent != member.better_than_any_parent
                or detailed_sha256 != member.detailed_evaluation_sha256
                or disposition is not member.disposition
                or candidate_failure != member.candidate_failure
            ):
                raise ValueError("outcome differs from its recombination receipt")

    @property
    def candidates(self) -> tuple[EvolutionCandidate, ...]:
        values: list[EvolutionCandidate] = []
        for outcome in self.outcomes:
            candidate = outcome.candidate
            if candidate is None:  # pragma: no cover - closed by __post_init__.
                raise AssertionError("validated outcome lost its candidate")
            values.append(candidate)
        return tuple(values)

    @property
    def scored_candidates(self) -> tuple[EvolutionCandidate, ...]:
        """Return selected unions with complete decision-objective vectors."""

        return tuple(
            candidate
            for member, candidate in zip(
                self.receipt.members,
                self.candidates,
                strict=True,
            )
            if member.disposition is PortfolioMemberDisposition.SCORED
        )

    @property
    def infeasible_candidates(self) -> tuple[EvolutionCandidate, ...]:
        """Return selected unions rejected by workload evaluation evidence."""

        return tuple(
            candidate
            for member, candidate in zip(
                self.receipt.members,
                self.candidates,
                strict=True,
            )
            if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
        )


@dataclass(slots=True)
class PortfolioRecombination:
    """Enumerate safe disjoint unions and evaluate exploit/coverage selections."""

    engine: MaterializedPortfolioEngine
    ids: IdFactory
    recombiner: DisjointPatchRecombiner = field(default_factory=DisjointPatchRecombiner)
    pair_policy: DisjointParentPairPolicy = field(
        default_factory=DisjointParentPairPolicy
    )
    archive_pair_policy: ArchiveAwareDisjointParentPairPolicy = field(
        default_factory=ArchiveAwareDisjointParentPairPolicy
    )
    selection_limit: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.engine, MaterializedPortfolioEngine):
            raise TypeError("engine must implement run_materialized_invocations")
        if not isinstance(self.ids, IdFactory):
            raise TypeError("ids must implement IdFactory")
        if type(self.recombiner) is not DisjointPatchRecombiner:
            raise TypeError("recombiner must be an exact DisjointPatchRecombiner")
        DisjointPatchRecombiner.__post_init__(self.recombiner)
        if type(self.pair_policy) is not DisjointParentPairPolicy:
            raise TypeError("pair_policy must be exact")
        if type(self.archive_pair_policy) is not ArchiveAwareDisjointParentPairPolicy:
            raise TypeError("archive_pair_policy must be exact")
        if type(self.selection_limit) is not int or self.selection_limit not in {1, 2}:
            raise ValueError("selection_limit must be one or two")

    @staticmethod
    def _branch_bindings(
        request: PortfolioRecombinationWaveRequest,
    ) -> tuple[
        tuple[PortfolioRecombinationBranchBinding, ...],
        tuple[PortfolioRecombinationSourceExclusionReceipt, ...],
    ]:
        contract = request.source_wave.selection_request.finite_variation_contract
        exposures = {
            (_path_text(value.path), value.family): value.count
            for value in request.path_family_exposures
        }
        values: list[PortfolioRecombinationBranchBinding] = []
        exclusions: list[PortfolioRecombinationSourceExclusionReceipt] = []
        for member in request.source_result.receipt.members:
            materialization = member.materialization
            option = contract.resolve(materialization.option_id)
            if member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE:
                failure = member.candidate_failure
                if type(failure) is not PortfolioCandidateFailureEvidence:
                    raise AssertionError(
                        "validated infeasible source lost candidate failure evidence"
                    )
                exclusions.append(
                    PortfolioRecombinationSourceExclusionReceipt(
                        rank=materialization.rank,
                        candidate_id=materialization.candidate_id,
                        candidate_configuration_sha256=(
                            materialization.child_configuration_sha256
                        ),
                        option_id=materialization.option_id,
                        option_identity_sha256=(
                            materialization.option_identity_sha256
                        ),
                        family=option.family,
                        source_outcome_sha256=member.outcome_sha256,
                        candidate_failure=failure,
                    )
                )
                continue
            exposure = sum(
                exposures.get((path, option.family), 0)
                for path in materialization.changed_paths
            )
            values.append(
                PortfolioRecombinationBranchBinding(
                    rank=materialization.rank,
                    candidate_id=materialization.candidate_id,
                    candidate_configuration_sha256=(
                        materialization.child_configuration_sha256
                    ),
                    option_id=materialization.option_id,
                    option_identity_sha256=materialization.option_identity_sha256,
                    family=option.family,
                    role=f"rank_{materialization.rank:04d}",
                    reward=member.reward,
                    changed_paths=materialization.changed_paths,
                    path_family_exposure=exposure,
                    source_outcome_sha256=member.outcome_sha256,
                )
            )
        return tuple(values), tuple(exclusions)

    def _enumerate_pairs(
        self,
        request: PortfolioRecombinationWaveRequest,
        branches: tuple[PortfolioRecombinationBranchBinding, ...],
    ) -> tuple[
        tuple[PortfolioPairAttemptReceipt, ...],
        tuple[ReplayVerifiedDisjointPair, ...],
        dict[tuple[CandidateId, CandidateId], DisjointPatchMaterialization],
    ]:
        candidate_by_id = {
            value.candidate_id: value
            for value in request.source_result.scored_candidates
        }
        branch_by_id = {value.candidate_id: value for value in branches}
        source_ids = set(candidate_by_id)
        allocated_ids: set[CandidateId] = set()
        attempts: list[PortfolioPairAttemptReceipt] = []
        eligible: list[ReplayVerifiedDisjointPair] = []
        materializations: dict[
            tuple[CandidateId, CandidateId], DisjointPatchMaterialization
        ] = {}
        for left_id, right_id in combinations(sorted(source_ids), 2):
            target_id = self.ids.new_candidate_id()
            if (
                target_id in source_ids
                or target_id == request.ancestor.candidate_id
                or (target_id in allocated_ids)
            ):
                raise ValueError("recombination target candidate IDs collide")
            allocated_ids.add(target_id)
            left = candidate_by_id[left_id]
            right = candidate_by_id[right_id]
            try:
                materialization = self.recombiner.materialize(
                    ancestor=request.ancestor.configuration,
                    ancestor_candidate_id=request.ancestor.candidate_id,
                    left=left.configuration,
                    left_candidate_id=left_id,
                    right=right.configuration,
                    right_candidate_id=right_id,
                    target_candidate_id=target_id,
                )
            except DisjointPatchRecombinationError:
                attempts.append(
                    PortfolioPairAttemptReceipt(
                        left_candidate_id=left_id,
                        right_candidate_id=right_id,
                        target_candidate_id=target_id,
                        replay_safe=False,
                    )
                )
                continue
            target_sha256 = typed_json_sha256(materialization.configuration)
            attempt = PortfolioPairAttemptReceipt(
                left_candidate_id=left_id,
                right_candidate_id=right_id,
                target_candidate_id=target_id,
                replay_safe=True,
                target_configuration_sha256=target_sha256,
                materialization_receipt_sha256=materialization.receipt_sha256,
                union_patch_sha256=materialization.union_patch.patch_hash,
            )
            pair = ReplayVerifiedDisjointPair(
                left=branch_by_id[left_id].facts,
                right=branch_by_id[right_id].facts,
                target_configuration_sha256=target_sha256,
                materialization_receipt_sha256=materialization.receipt_sha256,
            )
            attempts.append(attempt)
            eligible.append(pair)
            materializations[(left_id, right_id)] = materialization
        return tuple(attempts), tuple(eligible), materializations

    @staticmethod
    def _join_outcome(
        *,
        role: PortfolioRecombinationRole,
        invocation: MaterializedInvocation,
        materialization: DisjointPatchMaterialization,
        branches: dict[CandidateId, PortfolioRecombinationBranchBinding],
        outcome: InvocationOutcome,
    ) -> PortfolioRecombinationMemberReceipt:
        if type(outcome) is not InvocationOutcome:
            raise TypeError("engine outcomes must be exact InvocationOutcome values")
        InvocationOutcome.__post_init__(outcome)
        candidate = outcome.candidate
        prepared = outcome.prepared
        pair_ids = tuple(parent.candidate_id for parent in invocation.plan.parents)
        if len(pair_ids) != 2:
            raise AssertionError("recombination invocation lost its two parents")
        if (
            prepared.plan != invocation.plan
            or prepared.proposal_authority is not ProposalAuthority.ENGINE
            or prepared.call_id is not None
            or prepared.candidate_id != invocation.candidate_id
            or prepared.materialized_candidate_id != invocation.candidate_id
            or prepared.materialization_policy_id != materialization.policy_id
            or prepared.materialization_policy_version != materialization.policy_version
            or prepared.materialization_receipt_hash != materialization.receipt_sha256
        ):
            raise ValueError("engine outcome differs from materialized recombination")
        if (
            outcome.failure_stage is not None
            or candidate is None
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
        ):
            raise ValueError("recombination wave contains a failed or partial member")
        draft = invocation.draft
        expected_attribution = draft.source_attribution
        expected_parent_patches = tuple(
            derive_patch(
                parent.configuration,
                materialization.configuration,
                base_candidate_id=parent.candidate_id,
                target_candidate_id=invocation.candidate_id,
            ).patch_hash
            for parent in invocation.plan.parents
        )
        if (
            candidate.candidate_id != invocation.candidate_id
            or not typed_json_equal(
                candidate.configuration, materialization.configuration
            )
            or candidate.generation != invocation.plan.generation
            or candidate.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION
            or candidate.parent_ids != pair_ids
            or candidate.common_ancestor_id
            != materialization.classification.ancestor_candidate_id
            or candidate.parent_patch_hashes != expected_parent_patches
            or candidate.preservation_verified is not True
            or candidate.call_telemetry is not None
            or candidate.source_attribution != expected_attribution
        ):
            raise ValueError("candidate differs from exact disjoint materialization")
        detailed = candidate.detailed_evaluation
        disposition = PortfolioMemberDisposition.SCORED
        candidate_failure: PortfolioCandidateFailureEvidence | None = None
        if not candidate.valid:
            if detailed is None or detailed.failure is None:
                raise ValueError(
                    "candidate infeasibility requires detailed evaluator evidence"
                )
            candidate_failure = PortfolioCandidateFailureEvidence.from_failure_record(
                detailed.failure,
                detailed_evaluation_sha256=detailed.evidence_sha256,
            )
            disposition = PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
            if outcome.parent_relations:
                raise ValueError(
                    "candidate infeasibility cannot publish parent relations"
                )
            if outcome.dominates_any_parent or outcome.better_than_any_parent:
                raise ValueError(
                    "candidate infeasibility cannot publish improvement flags"
                )
        left, right = (branches[value] for value in pair_ids)
        return PortfolioRecombinationMemberReceipt(
            selection_role=role,
            pair_ids=pair_ids,
            source_ranks=(left.rank, right.rank),
            source_option_ids=(left.option_id, right.option_id),
            source_families=(left.family, right.family),
            target_candidate_id=candidate.candidate_id,
            target_configuration_sha256=candidate.occurrence.configuration_hash,
            materialization_receipt_sha256=materialization.receipt_sha256,
            union_patch_sha256=materialization.union_patch.patch_hash,
            parent_patch_sha256s=expected_parent_patches,
            source_attribution=expected_attribution,
            operator_invocation_id=prepared.operator_invocation_id,
            reward_definition_sha256=prepared.variation_case.reward_definition_hash,
            reward=outcome.reward,
            parent_relations=outcome.parent_relations,
            detailed_evaluation_sha256=(
                None if detailed is None else detailed.evidence_sha256
            ),
            dominates_any_parent=outcome.dominates_any_parent,
            better_than_any_parent=outcome.better_than_any_parent,
            disposition=disposition,
            candidate_failure=candidate_failure,
        )

    async def run(
        self,
        request: PortfolioRecombinationWaveRequest,
        *,
        reward_binding: RewardPolicyBinding | None = None,
    ) -> PortfolioRecombinationWaveResult:
        """Enumerate all source pairs and evaluate a bounded deterministic slate.

        The pair policy can nominate an exploit pair and a coverage pair, while
        the enclosing campaign protocol owns the offspring envelope.  Keeping
        that envelope explicit here lets generic campaign shapes request one
        recombination child without evaluating and then discarding a second
        child outside their preregistered budget.
        """

        if type(request) is not PortfolioRecombinationWaveRequest:
            raise TypeError("request must be an exact recombination wave request")
        PortfolioRecombinationWaveRequest.__post_init__(request)
        if reward_binding is not None:
            if type(reward_binding) is not RewardPolicyBinding:
                raise TypeError("reward_binding must be exact or None")
            RewardPolicyBinding.__post_init__(reward_binding)
        source_receipt_sha256 = request.source_result.receipt.receipt_sha256
        source_request_sha256 = request.source_wave.selection_request.request_sha256
        source_contract_sha256 = (
            request.source_wave.selection_request.finite_variation_contract.identity_sha256
        )
        ancestor_sha256 = request.ancestor.occurrence.configuration_hash
        generation = request.generation
        label_prefix = request.label_prefix
        phase = request.phase
        exposure_snapshot = request.path_family_exposures
        source_archive_snapshot = request.source_archive_snapshot
        source_utilities = request.source_utilities
        source_archive_snapshot_sha256 = (
            None
            if source_archive_snapshot is None
            else source_archive_snapshot.snapshot_sha256
        )
        source_utilities_sha256 = (
            None if source_utilities is None else source_utilities.receipt_sha256
        )
        branches, source_exclusions = self._branch_bindings(request)
        attempts, eligible, materializations = self._enumerate_pairs(
            request,
            branches,
        )
        decision: PortfolioRecombinationPairDecision
        if source_utilities is None:
            decision = self.pair_policy.select(eligible)
        else:
            decision = self.archive_pair_policy.select(
                eligible,
                source_utilities=source_utilities,
            )
        selected: list[
            tuple[PortfolioRecombinationRole, tuple[CandidateId, CandidateId]]
        ] = []
        if decision.exploit_pair_ids is not None:
            selected.append(("exploit", decision.exploit_pair_ids))
        if decision.coverage_pair_ids is not None:
            selected.append(("coverage", decision.coverage_pair_ids))
        selected = selected[: self.selection_limit]
        candidate_by_id = {
            value.candidate_id: value
            for value in request.source_result.scored_candidates
        }
        invocations: list[MaterializedInvocation] = []
        selected_materializations: list[DisjointPatchMaterialization] = []
        for role, pair_ids in selected:
            materialization = materializations[pair_ids]
            invocation = materialized_disjoint_invocation(
                plan=InvocationPlan(
                    operator_kind=OperatorKind.THREE_WAY_RECOMBINATION,
                    parents=tuple(candidate_by_id[value] for value in pair_ids),
                    generation=generation,
                    label=f"{label_prefix}.{role}",
                    common_ancestor=request.ancestor,
                    phase=phase,
                ),
                materialization=materialization,
            )
            invocations.append(invocation)
            selected_materializations.append(materialization)
        invocation_tuple = tuple(invocations)
        if invocation_tuple:
            outcomes = await self.engine.run_materialized_invocations(
                invocation_tuple,
                reward_binding=reward_binding,
            )
        else:
            outcomes = ()
        if type(outcomes) is not tuple or len(outcomes) != len(invocation_tuple):
            raise ValueError("engine returned a partial recombination outcome wave")
        branch_by_id = {value.candidate_id: value for value in branches}
        members = tuple(
            self._join_outcome(
                role=role,
                invocation=invocation,
                materialization=materialization,
                branches=branch_by_id,
                outcome=outcome,
            )
            for (role, _), invocation, materialization, outcome in zip(
                selected,
                invocation_tuple,
                selected_materializations,
                outcomes,
                strict=True,
            )
        )
        if (
            request.generation != generation
            or request.label_prefix != label_prefix
            or request.phase != phase
            or request.path_family_exposures != exposure_snapshot
            or request.source_archive_snapshot != source_archive_snapshot
            or request.source_utilities != source_utilities
            or (
                None
                if request.source_archive_snapshot is None
                else request.source_archive_snapshot.snapshot_sha256
            )
            != source_archive_snapshot_sha256
            or (
                None
                if request.source_utilities is None
                else request.source_utilities.receipt_sha256
            )
            != source_utilities_sha256
            or request.source_result.receipt.receipt_sha256 != source_receipt_sha256
            or request.source_wave.selection_request.request_sha256
            != source_request_sha256
            or request.source_wave.selection_request.finite_variation_contract.identity_sha256
            != source_contract_sha256
            or request.ancestor.occurrence.configuration_hash != ancestor_sha256
        ):
            raise ValueError("source wave or ancestor drifted during recombination")
        no_pair = (
            None
            if eligible
            else PortfolioRecombinationNoPairReceipt(
                reason=(
                    PortfolioRecombinationNoPairReason.INSUFFICIENT_SCORED_SOURCES
                    if len(branches) < 2
                    else PortfolioRecombinationNoPairReason.NO_REPLAY_SAFE_DISJOINT_PAIR
                ),
                scored_source_count=len(branches),
                excluded_source_count=len(source_exclusions),
                enumerated_pair_count=len(attempts),
                replay_safe_pair_count=0,
            )
        )
        receipt = PortfolioRecombinationWaveReceipt(
            source_wave_receipt_sha256=source_receipt_sha256,
            source_request_sha256=source_request_sha256,
            source_decision_sha256=request.source_result.receipt.decision_sha256,
            source_contract_sha256=source_contract_sha256,
            ancestor_candidate_id=request.ancestor.candidate_id,
            ancestor_configuration_sha256=ancestor_sha256,
            generation=generation,
            branches=branches,
            source_exclusions=source_exclusions,
            path_family_exposures=exposure_snapshot,
            pair_attempts=attempts,
            pair_decision=decision,
            members=members,
            selection_limit=self.selection_limit,
            no_pair=no_pair,
        )
        return PortfolioRecombinationWaveResult(receipt=receipt, outcomes=outcomes)


__all__ = [
    "ArchiveAwareDisjointPairSelectionDecision",
    "ArchiveAwareDisjointParentPairPolicy",
    "FrozenArchiveBranchUtility",
    "FrozenArchiveSourcePairUtility",
    "FrozenArchiveSourceUtilityContext",
    "FrozenArchiveSourceUtilityReceipt",
    "ObservedSourceBranch",
    "PortfolioPairAttemptReceipt",
    "PortfolioRecombination",
    "PortfolioRecombinationBranchBinding",
    "PortfolioRecombinationMemberReceipt",
    "PortfolioRecombinationNoPairReason",
    "PortfolioRecombinationNoPairReceipt",
    "PortfolioRecombinationSourceExclusionReason",
    "PortfolioRecombinationSourceExclusionReceipt",
    "PortfolioRecombinationWaveReceipt",
    "PortfolioRecombinationWaveRequest",
    "PortfolioRecombinationWaveResult",
    "bind_portfolio_recombination_source_utilities",
    "frozen_archive_source_utility_context",
    "portfolio_recombination_observed_sources",
]
