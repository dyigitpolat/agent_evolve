"""Opt-in prior-calibrated allocation over an eight-member finite slate.

The provider-facing schema is intentionally outside this module.  A caller
supplies sealed prediction receipts, structural evidence, and an immutable
prior-wave calibration snapshot.  The legacy model top-k prefix remains the
default; calibrated mode assigns exactly four distinct engine-owned roles.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from itertools import permutations
from typing import ClassVar

from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationCell,
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseAssessment,
    PortfolioMemoryDoseMember,
    PortfolioMemoryDoseStage,
    assess_evaluated_portfolio_memory_dose,
    assess_proposed_portfolio_memory_dose,
)


POLICY_ID = "trace_calibrated_four_role_slate"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:trace-calibrated-four-role-slate:v1;"
    b"legacy-top-k-default=true;calibrated-mode-opt-in=true;"
    b"calibration-cutoff-exclusive=true;beta-prior=true;"
    b"slate-size=8;portfolio-size=4;"
    b"roles=calibrated-exploit,memory-hypothesis,falsification,coverage;"
    b"benchmark-owned-meaningful-direction=true;"
    b"joint-exact-assignment=true"
).hexdigest()

_STRUCTURAL_DOMAIN = b"agent-evolve:slate-structural-evidence:v1\x00"
_SLATE_DOMAIN = b"agent-evolve:calibrated-slate:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:slate-allocation-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:slate-allocation-decision:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_OPTION = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_MAX_WAVE = (1 << 63) - 1
_MAX_SLATE_SIZE = 8


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


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _require_metric(value: str, *, name: str = "metric_id") -> None:
    if type(value) is not str or _METRIC.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed metric identifier grammar")


def _require_option(value: str, *, name: str = "option_id") -> None:
    if type(value) is not str or _OPTION.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed option identifier grammar")


def _require_wave(value: int, *, name: str) -> None:
    if type(value) is not int or not 1 <= value <= _MAX_WAVE:
        raise ValueError(f"{name} must be an exact positive int63")


def _require_finite_float(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


def _require_unit_interval(value: float, *, name: str) -> None:
    _require_finite_float(value, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _canonical_tokens(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _TOKEN.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of closed tokens")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


class SlateRoleProposal(str, Enum):
    """Untrusted semantic role proposed by the model."""

    EXPLOIT = "exploit"
    FALSIFY = "falsify"
    COVERAGE = "coverage"


class MetricOptimizationGoal(str, Enum):
    """Direction in which an objective improves."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class SlateAllocationMode(str, Enum):
    """The legacy path is deliberately the default policy mode."""

    DIRECT_MODEL_TOP_K = "direct_model_top_k"
    CALIBRATED_FOUR_ROLE = "calibrated_four_role"


class SlateAllocationRole(str, Enum):
    """Engine-owned role attached to one evaluated slate member."""

    DIRECT_MODEL_TOP_K = "direct_model_top_k"
    CALIBRATED_EXPLOIT = "calibrated_exploit"
    MEMORY_HYPOTHESIS = "memory_hypothesis"
    FALSIFICATION_DISAGREEMENT = "falsification_disagreement"
    STRUCTURAL_COVERAGE = "structural_coverage"


_CALIBRATED_ROLES = (
    SlateAllocationRole.CALIBRATED_EXPLOIT,
    SlateAllocationRole.MEMORY_HYPOTHESIS,
    SlateAllocationRole.FALSIFICATION_DISAGREEMENT,
    SlateAllocationRole.STRUCTURAL_COVERAGE,
)


@dataclass(frozen=True, slots=True)
class SlateStructuralEvidence:
    """Benchmark-injected normalized novelty and structural-coverage evidence."""

    frozen_archive_snapshot_sha256: str
    evidence_receipt_sha256: str
    archive_novelty_score: float
    structural_coverage_score: float

    def __post_init__(self) -> None:
        require_sha256(
            self.frozen_archive_snapshot_sha256,
            "frozen_archive_snapshot_sha256",
        )
        require_sha256(self.evidence_receipt_sha256, "evidence_receipt_sha256")
        _require_unit_interval(
            self.archive_novelty_score,
            name="archive_novelty_score",
        )
        _require_unit_interval(
            self.structural_coverage_score,
            name="structural_coverage_score",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "frozen_archive_snapshot_sha256": (self.frozen_archive_snapshot_sha256),
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "archive_novelty_score_hex": self.archive_novelty_score.hex(),
            "structural_coverage_score_hex": (self.structural_coverage_score.hex()),
        }

    @property
    def evidence_sha256(self) -> str:
        return _hash(_STRUCTURAL_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class CalibratedSlateMember:
    """One sealed finite option in the model-proposed K-slate."""

    model_rank: int
    option_id: str
    option_identity_sha256: str
    family: str
    locus_key: str
    phenotype_identity_sha256: str
    supporting_card_keys: tuple[str, ...]
    role_proposal: SlateRoleProposal
    rationale_sha256: str
    predictions: tuple[ForecastPredictionReceipt, ...]
    structural_evidence: SlateStructuralEvidence

    def __post_init__(self) -> None:
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        _require_option(self.option_id)
        for name in (
            "option_identity_sha256",
            "phenotype_identity_sha256",
            "rationale_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_token(self.family, name="family")
        _require_token(self.locus_key, name="locus_key")
        _canonical_tokens(self.supporting_card_keys, name="supporting_card_keys")
        if type(self.role_proposal) is not SlateRoleProposal:
            raise TypeError("role_proposal must be exact SlateRoleProposal")
        if (
            type(self.predictions) is not tuple
            or not self.predictions
            or any(
                type(value) is not ForecastPredictionReceipt
                for value in self.predictions
            )
        ):
            raise ValueError("predictions must contain exact prediction receipts")
        for value in self.predictions:
            value.revalidate()
        if tuple(value.metric_id for value in self.predictions) != tuple(
            sorted({value.metric_id for value in self.predictions})
        ):
            raise ValueError("predictions must have unique canonical metric order")
        for value in self.predictions:
            if (
                value.option_id != self.option_id
                or value.option_identity_sha256 != self.option_identity_sha256
                or value.family != self.family
            ):
                raise ValueError("prediction receipt belongs to a foreign slate member")
        if type(self.structural_evidence) is not SlateStructuralEvidence:
            raise TypeError("structural_evidence must be exact SlateStructuralEvidence")
        self.structural_evidence.__post_init__()

    def revalidate(self) -> None:
        if type(self) is not CalibratedSlateMember:
            raise TypeError("member must be exact CalibratedSlateMember")
        CalibratedSlateMember.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "model_rank": self.model_rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "locus_key": self.locus_key,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "supporting_card_keys": list(self.supporting_card_keys),
            "role_proposal": self.role_proposal.value,
            "rationale_sha256": self.rationale_sha256,
            "predictions": [value.to_record() for value in self.predictions],
            "structural_evidence": self.structural_evidence.to_record(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class CalibratedSlate:
    """Exact K-member proposal bound to one parent and selector decision."""

    scope: ForecastCalibrationScope
    wave_index: int
    selector_decision_sha256: str
    parent_candidate_identity_sha256: str
    finite_contract_sha256: str
    members: tuple[CalibratedSlateMember, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        _require_wave(self.wave_index, name="wave_index")
        for name in (
            "selector_decision_sha256",
            "parent_candidate_identity_sha256",
            "finite_contract_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.members) is not tuple
            or not 1 <= len(self.members) <= _MAX_SLATE_SIZE
            or any(type(value) is not CalibratedSlateMember for value in self.members)
        ):
            raise ValueError("members must be a bounded exact slate tuple")
        for value in self.members:
            value.revalidate()
        if tuple(value.model_rank for value in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("slate members must preserve contiguous model rank")
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("slate option IDs must be distinct")
        if len({value.option_identity_sha256 for value in self.members}) != len(
            self.members
        ):
            raise ValueError("slate option identities must be distinct")
        archive_snapshots = {
            value.structural_evidence.frozen_archive_snapshot_sha256
            for value in self.members
        }
        if len(archive_snapshots) != 1:
            raise ValueError("structural evidence must share one frozen archive")
        for member in self.members:
            for prediction in member.predictions:
                if (
                    prediction.scope != self.scope
                    or prediction.wave_index != self.wave_index
                    or prediction.selector_decision_sha256
                    != self.selector_decision_sha256
                    or prediction.parent_candidate_identity_sha256
                    != self.parent_candidate_identity_sha256
                ):
                    raise ValueError("slate contains a foreign prediction receipt")

    def revalidate(self) -> None:
        if type(self) is not CalibratedSlate:
            raise TypeError("slate must be exact CalibratedSlate")
        CalibratedSlate.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "scope_sha256": self.scope.scope_sha256,
            "wave_index": self.wave_index,
            "selector_decision_sha256": self.selector_decision_sha256,
            "parent_candidate_identity_sha256": (self.parent_candidate_identity_sha256),
            "finite_contract_sha256": self.finite_contract_sha256,
            "members": [value.to_record() for value in self.members],
        }

    @property
    def slate_sha256(self) -> str:
        return _hash(_SLATE_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "slate_sha256": self.slate_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is CalibratedSlate and self.slate_sha256 == other.slate_sha256
        )

    __hash__ = None


def _memory_dose_members(
    members: tuple[CalibratedSlateMember, ...],
) -> tuple[PortfolioMemoryDoseMember, ...]:
    return tuple(
        PortfolioMemoryDoseMember(
            rank=rank,
            option_id=value.option_id,
            option_identity_sha256=value.option_identity_sha256,
            supporting_card_keys=value.supporting_card_keys,
        )
        for rank, value in enumerate(members, start=1)
    )


@dataclass(frozen=True, slots=True)
class SlateMetricObjective:
    """Benchmark-injected interpretation of one required prediction metric."""

    metric_id: str
    goal: MetricOptimizationGoal
    weight: float
    definition_sha256: str

    def __post_init__(self) -> None:
        _require_metric(self.metric_id)
        if type(self.goal) is not MetricOptimizationGoal:
            raise TypeError("goal must be exact MetricOptimizationGoal")
        _require_finite_float(self.weight, name="weight")
        if self.weight <= 0.0:
            raise ValueError("weight must be strictly positive")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal.value,
            "weight_hex": self.weight.hex(),
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class SlateAllocationRequest:
    """Complete provider-free inputs for legacy or calibrated allocation."""

    slate: CalibratedSlate
    portfolio_size: int
    objectives: tuple[SlateMetricObjective, ...]
    assigned_card_keys: tuple[str, ...]
    calibration_snapshot: ForecastCalibrationSnapshot | None = None
    pairwise_disjoint_option_id_pairs: tuple[tuple[str, str], ...] | None = None
    min_distinct_families: int | None = None
    memory_dose_contract: BoundedPortfolioMemoryDoseContract | None = None
    proposal_memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None
    required_option_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.slate) is not CalibratedSlate:
            raise TypeError("slate must be exact CalibratedSlate")
        self.slate.revalidate()
        if type(self.portfolio_size) is not int or not 1 <= self.portfolio_size <= len(
            self.slate.members
        ):
            raise ValueError("portfolio_size must lie within the finite slate")
        if (
            type(self.objectives) is not tuple
            or not self.objectives
            or any(type(value) is not SlateMetricObjective for value in self.objectives)
        ):
            raise ValueError("objectives must contain exact metric objectives")
        for value in self.objectives:
            value.__post_init__()
        objective_ids = tuple(value.metric_id for value in self.objectives)
        if objective_ids != tuple(sorted(set(objective_ids))):
            raise ValueError("objectives must have unique canonical metric order")
        if any(
            tuple(value.metric_id for value in member.predictions) != objective_ids
            for member in self.slate.members
        ):
            raise ValueError("every slate member must predict every objective once")
        _canonical_tokens(self.assigned_card_keys, name="assigned_card_keys")
        if self.calibration_snapshot is not None:
            if type(self.calibration_snapshot) is not ForecastCalibrationSnapshot:
                raise TypeError(
                    "calibration_snapshot must be exact ForecastCalibrationSnapshot"
                )
            self.calibration_snapshot.revalidate()
            if self.calibration_snapshot.scope != self.slate.scope:
                raise ValueError("calibration snapshot has a foreign scope")
        pairs = self.pairwise_disjoint_option_id_pairs
        if pairs is not None:
            if type(pairs) is not tuple or any(
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(value) is not str for value in pair)
                for pair in pairs
            ):
                raise TypeError(
                    "pairwise_disjoint_option_id_pairs must be exact pairs or None"
                )
            option_ids = {value.option_id for value in self.slate.members}
            for left, right in pairs:
                _require_option(left, name="pairwise_disjoint_option_id_pairs.left")
                _require_option(right, name="pairwise_disjoint_option_id_pairs.right")
                if left >= right:
                    raise ValueError("disjoint option pairs must be canonical")
                if left not in option_ids or right not in option_ids:
                    raise ValueError("disjoint option pair escapes the slate")
            if pairs != tuple(sorted(set(pairs))):
                raise ValueError("disjoint option pairs must be unique and canonical")
        if self.min_distinct_families is not None:
            if (
                type(self.min_distinct_families) is not int
                or not 1 <= self.min_distinct_families <= self.portfolio_size
            ):
                raise ValueError(
                    "min_distinct_families must lie within the portfolio size"
                )
            if self.min_distinct_families > len(
                {value.family for value in self.slate.members}
            ):
                raise ValueError("slate cannot satisfy min_distinct_families")
        if type(self.required_option_ids) is not tuple or any(
            type(value) is not str for value in self.required_option_ids
        ):
            raise TypeError("required_option_ids must be an exact string tuple")
        if self.required_option_ids != tuple(
            sorted(set(self.required_option_ids))
        ):
            raise ValueError("required_option_ids must be unique and canonical")
        slate_option_ids = {value.option_id for value in self.slate.members}
        if not set(self.required_option_ids).issubset(slate_option_ids):
            raise ValueError("required_option_ids escape the sealed slate")
        if len(self.required_option_ids) > self.portfolio_size:
            raise ValueError("required_option_ids exceed the portfolio size")
        if (self.memory_dose_contract is None) != (
            self.proposal_memory_dose_assessment is None
        ):
            raise ValueError(
                "memory-dose contract and proposal assessment must be supplied together"
            )
        if self.memory_dose_contract is not None:
            if type(self.memory_dose_contract) is not (
                BoundedPortfolioMemoryDoseContract
            ):
                raise TypeError("memory_dose_contract must be exact or None")
            if type(self.proposal_memory_dose_assessment) is not (
                PortfolioMemoryDoseAssessment
            ):
                raise TypeError(
                    "proposal_memory_dose_assessment must be exact or None"
                )
            self.memory_dose_contract.__post_init__()
            self.proposal_memory_dose_assessment.__post_init__()
            if (
                self.memory_dose_contract.finite_contract_identity_sha256
                != self.slate.finite_contract_sha256
            ):
                raise ValueError("memory dose names a foreign finite contract")
            if (
                self.memory_dose_contract.assigned_card_keys
                != self.assigned_card_keys
            ):
                raise ValueError("memory-dose cards differ from assigned cards")
            expected_proposal = assess_proposed_portfolio_memory_dose(
                self.memory_dose_contract,
                _memory_dose_members(self.slate.members),
            )
            if (
                not expected_proposal.passed
                or self.proposal_memory_dose_assessment != expected_proposal
            ):
                raise ValueError(
                    "proposal memory-dose assessment differs from the sealed slate"
                )

    def revalidate(self) -> None:
        if type(self) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        SlateAllocationRequest.__post_init__(self)

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "slate": self.slate.to_record(),
            "portfolio_size": self.portfolio_size,
            "objectives": [value.to_record() for value in self.objectives],
            "assigned_card_keys": list(self.assigned_card_keys),
            "calibration_snapshot": (
                None
                if self.calibration_snapshot is None
                else self.calibration_snapshot.to_record()
            ),
            **(
                {}
                if self.pairwise_disjoint_option_id_pairs is None
                else {
                    "pairwise_disjoint_option_id_pairs": [
                        list(value)
                        for value in self.pairwise_disjoint_option_id_pairs
                    ]
                }
            ),
            **(
                {}
                if self.min_distinct_families is None
                else {"min_distinct_families": self.min_distinct_families}
            ),
            **(
                {}
                if self.memory_dose_contract is None
                else {
                    "memory_dose_contract": self.memory_dose_contract.to_record(),
                    "proposal_memory_dose_assessment": (
                        self.proposal_memory_dose_assessment.to_record()
                    ),
                }
            ),
            **(
                {}
                if not self.required_option_ids
                else {"required_option_ids": list(self.required_option_ids)}
            ),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is SlateAllocationRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


def assess_allocated_slate_memory_dose(
    request: SlateAllocationRequest,
    members: tuple[CalibratedSlateMember, ...],
) -> PortfolioMemoryDoseAssessment | None:
    """Assess one prospective K4 subset against the sealed K8 dose receipt."""

    if type(request) is not SlateAllocationRequest:
        raise TypeError("request must be exact SlateAllocationRequest")
    contract = request.memory_dose_contract
    proposal_assessment = request.proposal_memory_dose_assessment
    if (contract is None) != (proposal_assessment is None):
        raise ValueError(
            "memory-dose contract and proposal assessment must be supplied together"
        )
    if contract is None:
        return None
    if (
        type(members) is not tuple
        or len(members) != request.portfolio_size
        or any(type(value) is not CalibratedSlateMember for value in members)
    ):
        raise ValueError("members must be the exact allocated slate subset")
    slate_by_option = {value.option_id: value for value in request.slate.members}
    if any(
        value.option_id not in slate_by_option
        or slate_by_option[value.option_id] != value
        for value in members
    ):
        raise ValueError("allocated memory-dose member escapes the sealed slate")
    if type(contract) is not BoundedPortfolioMemoryDoseContract:
        raise TypeError("memory_dose_contract must be exact or None")
    if type(proposal_assessment) is not PortfolioMemoryDoseAssessment:
        raise TypeError("proposal_memory_dose_assessment must be exact or None")
    contract.__post_init__()
    proposal_assessment.__post_init__()
    if (
        contract.finite_contract_identity_sha256
        != request.slate.finite_contract_sha256
    ):
        raise ValueError("memory dose names a foreign finite contract")
    if contract.assigned_card_keys != request.assigned_card_keys:
        raise ValueError("memory-dose cards differ from assigned cards")
    if (
        proposal_assessment.stage is not PortfolioMemoryDoseStage.PROPOSED_SLATE
        or proposal_assessment.contract_sha256 != contract.contract_sha256
        or not proposal_assessment.passed
        or proposal_assessment.member_content_binding_sha256s
        != tuple(
            value.content_binding_sha256
            for value in _memory_dose_members(request.slate.members)
        )
    ):
        raise ValueError(
            "proposal memory-dose assessment differs from the sealed slate"
        )
    return assess_evaluated_portfolio_memory_dose(
        contract,
        _memory_dose_members(members),
        proposal_assessment=proposal_assessment,
    )


@dataclass(frozen=True, slots=True)
class MetricCalibrationAllocationScore:
    """Trace row explaining one metric's contribution to engine scores."""

    metric_id: str
    goal: MetricOptimizationGoal
    asserted_direction: MetricEffectDirection
    confidence: ForecastConfidenceBin
    weight: float
    calibration_cell: ForecastCalibrationCell
    calibration_source: str
    favorable_assertion: bool
    adverse_assertion: bool
    signed_exploitation_score: float
    falsification_score: float

    def __post_init__(self) -> None:
        _require_metric(self.metric_id)
        if type(self.goal) is not MetricOptimizationGoal:
            raise TypeError("goal must be exact MetricOptimizationGoal")
        if type(self.asserted_direction) is not MetricEffectDirection:
            raise TypeError("asserted_direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")
        _require_finite_float(self.weight, name="weight")
        if self.weight <= 0.0:
            raise ValueError("weight must be strictly positive")
        if type(self.calibration_cell) is not ForecastCalibrationCell:
            raise TypeError("calibration_cell must be exact ForecastCalibrationCell")
        self.calibration_cell.__post_init__()
        if self.calibration_source not in {
            "supported_family",
            "metric_direction_confidence",
            "declared_prior",
        }:
            raise ValueError("unsupported calibration_source")
        if type(self.favorable_assertion) is not bool:
            raise TypeError("favorable_assertion must be exact bool")
        if type(self.adverse_assertion) is not bool:
            raise TypeError("adverse_assertion must be exact bool")
        if self.favorable_assertion and self.adverse_assertion:
            raise ValueError("one assertion cannot be both favorable and adverse")
        _require_finite_float(
            self.signed_exploitation_score,
            name="signed_exploitation_score",
        )
        _require_finite_float(self.falsification_score, name="falsification_score")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal.value,
            "asserted_direction": self.asserted_direction.value,
            "confidence": self.confidence.value,
            "weight_hex": self.weight.hex(),
            "calibration_cell": self.calibration_cell.to_record(),
            "calibration_source": self.calibration_source,
            "favorable_assertion": self.favorable_assertion,
            "adverse_assertion": self.adverse_assertion,
            "signed_exploitation_score_hex": (self.signed_exploitation_score.hex()),
            "falsification_score_hex": self.falsification_score.hex(),
        }


@dataclass(frozen=True, slots=True)
class SlateMemberScoreRow:
    """All prior-only role scores for one member of the current slate."""

    option_id: str
    option_identity_sha256: str
    model_rank: int
    metric_scores: tuple[MetricCalibrationAllocationScore, ...]
    calibrated_exploitation_score: float
    memory_hypothesis_score: float
    falsification_disagreement_score: float
    structural_coverage_score: float
    supported_assigned_card_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_option(self.option_id)
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be positive")
        if (
            type(self.metric_scores) is not tuple
            or not self.metric_scores
            or any(
                type(value) is not MetricCalibrationAllocationScore
                for value in self.metric_scores
            )
        ):
            raise ValueError("metric_scores must contain exact metric score rows")
        for value in self.metric_scores:
            value.__post_init__()
        if tuple(value.metric_id for value in self.metric_scores) != tuple(
            sorted({value.metric_id for value in self.metric_scores})
        ):
            raise ValueError("metric_scores must use unique canonical metric order")
        for name in (
            "calibrated_exploitation_score",
            "memory_hypothesis_score",
            "falsification_disagreement_score",
            "structural_coverage_score",
        ):
            _require_finite_float(getattr(self, name), name=name)
        _canonical_tokens(
            self.supported_assigned_card_keys,
            name="supported_assigned_card_keys",
        )

    def score_for(self, role: SlateAllocationRole) -> float:
        self.__post_init__()
        if role is SlateAllocationRole.CALIBRATED_EXPLOIT:
            return self.calibrated_exploitation_score
        if role is SlateAllocationRole.MEMORY_HYPOTHESIS:
            return self.memory_hypothesis_score
        if role is SlateAllocationRole.FALSIFICATION_DISAGREEMENT:
            return self.falsification_disagreement_score
        if role is SlateAllocationRole.STRUCTURAL_COVERAGE:
            return self.structural_coverage_score
        raise ValueError("direct top-k has no calibrated role score")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "metric_scores": [value.to_record() for value in self.metric_scores],
            "calibrated_exploitation_score_hex": (
                self.calibrated_exploitation_score.hex()
            ),
            "memory_hypothesis_score_hex": self.memory_hypothesis_score.hex(),
            "falsification_disagreement_score_hex": (
                self.falsification_disagreement_score.hex()
            ),
            "structural_coverage_score_hex": self.structural_coverage_score.hex(),
            "supported_assigned_card_keys": list(self.supported_assigned_card_keys),
        }


def _favorable(
    direction: MetricEffectDirection,
    goal: MetricOptimizationGoal,
) -> tuple[bool, bool]:
    if direction in {MetricEffectDirection.UNKNOWN, MetricEffectDirection.UNCHANGED}:
        return False, False
    favorable = (
        goal is MetricOptimizationGoal.MINIMIZE
        and direction is MetricEffectDirection.DECREASE
    ) or (
        goal is MetricOptimizationGoal.MAXIMIZE
        and direction is MetricEffectDirection.INCREASE
    )
    return favorable, not favorable


def _score_member(
    request: SlateAllocationRequest,
    member: CalibratedSlateMember,
) -> SlateMemberScoreRow:
    snapshot = request.calibration_snapshot
    if snapshot is None:
        raise ValueError("calibrated allocation requires a calibration snapshot")
    objective_index = {value.metric_id: value for value in request.objectives}
    scores: list[MetricCalibrationAllocationScore] = []
    weighted_exploit = 0.0
    weighted_falsification = 0.0
    total_weight = sum(value.weight for value in request.objectives)
    for prediction in member.predictions:
        objective = objective_index[prediction.metric_id]
        cell, source = snapshot.lookup(
            metric_id=prediction.metric_id,
            asserted_direction=prediction.asserted_direction,
            confidence=prediction.confidence,
            family=member.family,
        )
        probability = cell.posterior_correctness
        favorable, adverse = _favorable(
            prediction.asserted_direction,
            objective.goal,
        )
        signed_exploit = probability if favorable else -probability if adverse else 0.0
        if prediction.asserted_direction is MetricEffectDirection.UNKNOWN:
            falsification = 0.0
        else:
            uncertainty = 1.0 - abs((2.0 * probability) - 1.0)
            calibrated_disagreement = max(0.0, 0.5 - probability) * 2.0
            falsification = (uncertainty + calibrated_disagreement) / 2.0
        weighted_exploit += objective.weight * signed_exploit
        weighted_falsification += objective.weight * falsification
        scores.append(
            MetricCalibrationAllocationScore(
                metric_id=prediction.metric_id,
                goal=objective.goal,
                asserted_direction=prediction.asserted_direction,
                confidence=prediction.confidence,
                weight=objective.weight,
                calibration_cell=cell,
                calibration_source=source,
                favorable_assertion=favorable,
                adverse_assertion=adverse,
                signed_exploitation_score=signed_exploit,
                falsification_score=falsification,
            )
        )
    exploit_score = weighted_exploit / total_weight
    falsification_score = weighted_falsification / total_weight
    supported = tuple(
        value
        for value in request.assigned_card_keys
        if value in member.supporting_card_keys
    )
    administration_fraction = len(supported) / len(request.assigned_card_keys)
    memory_score = exploit_score + administration_fraction
    structural = (
        member.structural_evidence.archive_novelty_score
        + member.structural_evidence.structural_coverage_score
    ) / 2.0
    return SlateMemberScoreRow(
        option_id=member.option_id,
        option_identity_sha256=member.option_identity_sha256,
        model_rank=member.model_rank,
        metric_scores=tuple(scores),
        calibrated_exploitation_score=exploit_score,
        memory_hypothesis_score=memory_score,
        falsification_disagreement_score=falsification_score,
        structural_coverage_score=structural,
        supported_assigned_card_keys=supported,
    )


@dataclass(frozen=True, slots=True)
class AllocatedSlateMember:
    """One selected member and its engine-owned portfolio role."""

    role: SlateAllocationRole
    option_id: str
    option_identity_sha256: str
    model_rank: int
    role_score: float | None

    def __post_init__(self) -> None:
        if type(self.role) is not SlateAllocationRole:
            raise TypeError("role must be exact SlateAllocationRole")
        _require_option(self.option_id)
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be positive")
        if self.role is SlateAllocationRole.DIRECT_MODEL_TOP_K:
            if self.role_score is not None:
                raise ValueError("legacy top-k members have no calibrated score")
        else:
            if self.role_score is None:
                raise ValueError("calibrated members require a role score")
            _require_finite_float(self.role_score, name="role_score")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "role": self.role.value,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "model_rank": self.model_rank,
            "role_score_hex": (
                None if self.role_score is None else self.role_score.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class _CalibratedAssignment:
    selected: tuple[AllocatedSlateMember, ...]
    joint_score: float
    diversity_score: float
    distinct_family_count: int
    distinct_locus_count: int
    distinct_phenotype_count: int
    administered_card_keys: tuple[str, ...]
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None


def _best_calibrated_assignment(
    request: SlateAllocationRequest,
    score_rows: tuple[SlateMemberScoreRow, ...],
) -> _CalibratedAssignment:
    member_by_id = {value.option_id: value for value in request.slate.members}
    best_key: tuple[object, ...] | None = None
    best_rows: tuple[SlateMemberScoreRow, ...] | None = None
    best_role_scores: tuple[float, ...] | None = None
    best_joint_score: float | None = None
    best_diversity: float | None = None
    best_family_count: int | None = None
    best_locus_count: int | None = None
    best_phenotype_count: int | None = None
    best_administered: tuple[str, ...] | None = None
    memory_dose_pass_by_subset: dict[tuple[str, ...], bool] = {}
    compatible_pairs = (
        None
        if request.pairwise_disjoint_option_id_pairs is None
        else {frozenset(value) for value in request.pairwise_disjoint_option_id_pairs}
    )
    for rows in permutations(score_rows, 4):
        if compatible_pairs is not None and any(
            frozenset((left.option_id, right.option_id)) not in compatible_pairs
            for left_index, left in enumerate(rows)
            for right in rows[left_index + 1 :]
        ):
            continue
        if request.min_distinct_families is not None and len(
            {member_by_id[row.option_id].family for row in rows}
        ) < request.min_distinct_families:
            continue
        memory_row = rows[1]
        if not memory_row.supported_assigned_card_keys:
            continue
        administered = tuple(
            sorted({card for row in rows for card in row.supported_assigned_card_keys})
        )
        if administered != request.assigned_card_keys:
            continue
        members = tuple(member_by_id[row.option_id] for row in rows)
        if request.memory_dose_contract is not None:
            dose_subset_key = tuple(sorted(row.option_id for row in rows))
            dose_passed = memory_dose_pass_by_subset.get(dose_subset_key)
            if dose_passed is None:
                canonical_members = tuple(
                    member_by_id[option_id] for option_id in dose_subset_key
                )
                dose_passed = assess_allocated_slate_memory_dose(
                    request,
                    canonical_members,
                ).passed
                memory_dose_pass_by_subset[dose_subset_key] = dose_passed
            if not dose_passed:
                continue
        family_count = len({value.family for value in members})
        locus_count = len({value.locus_key for value in members})
        phenotype_count = len({value.phenotype_identity_sha256 for value in members})
        diversity = (family_count + locus_count + phenotype_count) / 12.0
        role_scores = tuple(
            row.score_for(role) for role, row in zip(_CALIBRATED_ROLES, rows)
        )
        joint_score = sum(role_scores) + diversity
        tie_key: tuple[object, ...] = (
            -joint_score,
            *(-value for value in role_scores),
            tuple(value.option_id for value in rows),
        )
        if best_key is None or tie_key < best_key:
            best_key = tie_key
            best_rows = rows
            best_role_scores = role_scores
            best_joint_score = joint_score
            best_diversity = diversity
            best_family_count = family_count
            best_locus_count = locus_count
            best_phenotype_count = phenotype_count
            best_administered = administered
    if best_rows is None:
        raise ValueError(
            "slate has no four-role assignment administering every assigned card"
        )
    assert best_role_scores is not None
    assert best_joint_score is not None
    assert best_diversity is not None
    assert best_family_count is not None
    assert best_locus_count is not None
    assert best_phenotype_count is not None
    assert best_administered is not None
    selected = tuple(
        AllocatedSlateMember(
            role=role,
            option_id=row.option_id,
            option_identity_sha256=row.option_identity_sha256,
            model_rank=row.model_rank,
            role_score=score,
        )
        for role, row, score in zip(
            _CALIBRATED_ROLES,
            best_rows,
            best_role_scores,
        )
    )
    winner = _CalibratedAssignment(
        selected=selected,
        joint_score=best_joint_score,
        diversity_score=best_diversity,
        distinct_family_count=best_family_count,
        distinct_locus_count=best_locus_count,
        distinct_phenotype_count=best_phenotype_count,
        administered_card_keys=best_administered,
        memory_dose_assessment=None,
    )
    if request.memory_dose_contract is None:
        return winner
    winner_members = tuple(
        member_by_id[value.option_id] for value in winner.selected
    )
    exact_assessment = assess_allocated_slate_memory_dose(
        request,
        winner_members,
    )
    if not exact_assessment.passed:  # Defensive against cache/key drift.
        raise AssertionError("winning assignment violated bounded memory dose")
    return replace(winner, memory_dose_assessment=exact_assessment)


@dataclass(frozen=True, slots=True, eq=False)
class SlateAllocationDecision:
    """Replayable legacy top-k or prior-only calibrated allocation receipt."""

    request: SlateAllocationRequest
    mode: SlateAllocationMode
    score_rows: tuple[SlateMemberScoreRow, ...]
    selected: tuple[AllocatedSlateMember, ...]
    joint_score: float | None
    diversity_score: float | None
    distinct_family_count: int | None
    distinct_locus_count: int | None
    distinct_phenotype_count: int | None
    administered_card_keys: tuple[str, ...]
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        self.request.revalidate()
        if type(self.mode) is not SlateAllocationMode:
            raise TypeError("mode must be exact SlateAllocationMode")
        if type(self.score_rows) is not tuple or any(
            type(value) is not SlateMemberScoreRow for value in self.score_rows
        ):
            raise TypeError("score_rows must contain exact member scores")
        for value in self.score_rows:
            value.__post_init__()
        if type(self.selected) is not tuple or any(
            type(value) is not AllocatedSlateMember for value in self.selected
        ):
            raise TypeError("selected must contain exact allocated members")
        for value in self.selected:
            value.__post_init__()
        _canonical_tokens(self.administered_card_keys, name="administered_card_keys")
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not (
                PortfolioMemoryDoseAssessment
            ):
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()

        if self.mode is SlateAllocationMode.DIRECT_MODEL_TOP_K:
            expected_members = self.request.slate.members[: self.request.portfolio_size]
            expected = tuple(
                AllocatedSlateMember(
                    role=SlateAllocationRole.DIRECT_MODEL_TOP_K,
                    option_id=value.option_id,
                    option_identity_sha256=value.option_identity_sha256,
                    model_rank=value.model_rank,
                    role_score=None,
                )
                for value in expected_members
            )
            if self.score_rows or self.selected != expected:
                raise ValueError("legacy decision is not the exact model top-k prefix")
            expected_dose = (
                None
                if self.request.memory_dose_contract is None
                else assess_allocated_slate_memory_dose(
                    self.request,
                    expected_members,
                )
            )
            if expected_dose is not None and not expected_dose.passed:
                raise ValueError("direct model top-k violates bounded memory dose")
            expected_administered = (
                ()
                if expected_dose is None
                else self.request.assigned_card_keys
            )
            if (
                any(
                    value is not None
                    for value in (
                        self.joint_score,
                        self.diversity_score,
                        self.distinct_family_count,
                        self.distinct_locus_count,
                        self.distinct_phenotype_count,
                    )
                )
                or self.administered_card_keys != expected_administered
                or self.memory_dose_assessment != expected_dose
            ):
                raise ValueError("legacy top-k cannot claim calibrated evidence")
            return

        snapshot = self.request.calibration_snapshot
        if self.request.portfolio_size != 4:
            raise ValueError("calibrated mode requires exactly four evaluations")
        if len(self.request.slate.members) != _MAX_SLATE_SIZE:
            raise ValueError("calibrated mode requires the declared eight-member slate")
        if snapshot is None:
            raise ValueError("calibrated mode requires a prior snapshot")
        if snapshot.cutoff_wave_index_exclusive > self.request.slate.wave_index:
            raise ValueError("calibration snapshot cutoff reaches beyond current wave")
        if not self.request.assigned_card_keys:
            raise ValueError("memory role requires assigned card keys")
        expected_rows = tuple(
            _score_member(self.request, value) for value in self.request.slate.members
        )
        if self.score_rows != expected_rows:
            raise ValueError("score rows differ from prior calibration evidence")
        expected_assignment = _best_calibrated_assignment(
            self.request,
            expected_rows,
        )
        if self.selected != expected_assignment.selected:
            raise ValueError("selected roles do not replay the exact allocator")
        observed_summary = (
            self.joint_score,
            self.diversity_score,
            self.distinct_family_count,
            self.distinct_locus_count,
            self.distinct_phenotype_count,
            self.administered_card_keys,
            self.memory_dose_assessment,
        )
        expected_summary = (
            expected_assignment.joint_score,
            expected_assignment.diversity_score,
            expected_assignment.distinct_family_count,
            expected_assignment.distinct_locus_count,
            expected_assignment.distinct_phenotype_count,
            expected_assignment.administered_card_keys,
            expected_assignment.memory_dose_assessment,
        )
        if observed_summary != expected_summary:
            raise ValueError("allocation summary differs from exact assignment")

    def revalidate(self) -> None:
        if type(self) is not SlateAllocationDecision:
            raise TypeError("decision must be exact SlateAllocationDecision")
        SlateAllocationDecision.__post_init__(self)

    @property
    def prior_only(self) -> bool:
        self.revalidate()
        if self.mode is SlateAllocationMode.DIRECT_MODEL_TOP_K:
            return False
        snapshot = self.request.calibration_snapshot
        if snapshot is None:  # Defensive after validation.
            return False
        return all(
            value.prediction.wave_index < self.request.slate.wave_index
            for value in snapshot.observations
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "event_type": "trace_calibrated_slate_allocated",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "mode": self.mode.value,
            "request": self.request.to_record(),
            "request_sha256": self.request.request_sha256,
            "calibration_snapshot_sha256": (
                None
                if self.request.calibration_snapshot is None
                else self.request.calibration_snapshot.snapshot_sha256
            ),
            "prior_only": self.prior_only,
            "score_rows": [value.to_record() for value in self.score_rows],
            "selected": [value.to_record() for value in self.selected],
            "joint_score_hex": (
                None if self.joint_score is None else self.joint_score.hex()
            ),
            "diversity_score_hex": (
                None if self.diversity_score is None else self.diversity_score.hex()
            ),
            "distinct_family_count": self.distinct_family_count,
            "distinct_locus_count": self.distinct_locus_count,
            "distinct_phenotype_count": self.distinct_phenotype_count,
            "administered_card_keys": list(self.administered_card_keys),
            **(
                {}
                if self.memory_dose_assessment is None
                else {
                    "memory_dose_assessment": (
                        self.memory_dose_assessment.to_record()
                    )
                }
            ),
            "claim_scope": (
                "replayable_prior_only_allocation_not_efficacy_or_outcome_claim"
            ),
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is SlateAllocationDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class TraceCalibratedSlatePolicy:
    """Select the legacy prefix by default or opt into four-role allocation."""

    mode: SlateAllocationMode = SlateAllocationMode.DIRECT_MODEL_TOP_K

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION
    definition_sha256: ClassVar[str] = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.mode) is not SlateAllocationMode:
            raise TypeError("mode must be exact SlateAllocationMode")

    def select(self, request: SlateAllocationRequest) -> SlateAllocationDecision:
        if type(request) is not SlateAllocationRequest:
            raise TypeError("request must be exact SlateAllocationRequest")
        request.revalidate()
        if self.mode is SlateAllocationMode.DIRECT_MODEL_TOP_K:
            selected_members = request.slate.members[: request.portfolio_size]
            memory_dose_assessment = (
                None
                if request.memory_dose_contract is None
                else assess_allocated_slate_memory_dose(
                    request,
                    selected_members,
                )
            )
            if (
                memory_dose_assessment is not None
                and not memory_dose_assessment.passed
            ):
                raise ValueError("direct model top-k violates bounded memory dose")
            selected = tuple(
                AllocatedSlateMember(
                    role=SlateAllocationRole.DIRECT_MODEL_TOP_K,
                    option_id=value.option_id,
                    option_identity_sha256=value.option_identity_sha256,
                    model_rank=value.model_rank,
                    role_score=None,
                )
                for value in selected_members
            )
            return SlateAllocationDecision(
                request=request,
                mode=self.mode,
                score_rows=(),
                selected=selected,
                joint_score=None,
                diversity_score=None,
                distinct_family_count=None,
                distinct_locus_count=None,
                distinct_phenotype_count=None,
                administered_card_keys=(
                    ()
                    if memory_dose_assessment is None
                    else request.assigned_card_keys
                ),
                memory_dose_assessment=memory_dose_assessment,
            )
        rows = tuple(_score_member(request, value) for value in request.slate.members)
        assignment = _best_calibrated_assignment(request, rows)
        return SlateAllocationDecision(
            request=request,
            mode=self.mode,
            score_rows=rows,
            selected=assignment.selected,
            joint_score=assignment.joint_score,
            diversity_score=assignment.diversity_score,
            distinct_family_count=assignment.distinct_family_count,
            distinct_locus_count=assignment.distinct_locus_count,
            distinct_phenotype_count=assignment.distinct_phenotype_count,
            administered_card_keys=assignment.administered_card_keys,
            memory_dose_assessment=assignment.memory_dose_assessment,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "mode": self.mode.value,
        }


__all__ = [
    "CalibratedSlate",
    "CalibratedSlateMember",
    "MetricOptimizationGoal",
    "SlateAllocationDecision",
    "SlateAllocationMode",
    "SlateAllocationRequest",
    "SlateAllocationRole",
    "SlateMetricObjective",
    "SlateRoleProposal",
    "SlateStructuralEvidence",
    "TraceCalibratedSlatePolicy",
    "assess_allocated_slate_memory_dose",
]
