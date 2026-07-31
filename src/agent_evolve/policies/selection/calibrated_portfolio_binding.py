"""Authenticated inner-policy inputs for calibrated portfolio allocation.

The provider adapter must freeze this binding before it sends a prompt.  It
therefore cannot choose structural evidence after observing the model's slate,
and the model cannot author calibration, objective, or archive facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_slate import (
    SlateMetricObjective,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
)
from agent_evolve.policies.selection.proposal_support import (
    ProposalSupportCandidate,
    ProposalSupportDecision,
    StructuralProposalSupportPolicy,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjection,
)
from agent_evolve.policies.selection.common_candidate_pool import (
    CommonCandidatePoolDecision,
    TaskKeyedCommonCandidatePoolPolicy,
)
from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
    CompositionSelectionExposure,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    required_source_evaluation_option_ids,
)
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
)
from agent_evolve.ports.variation_source import (
    finite_variation_candidate_pool_required_option_ids,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_BINDING_DOMAIN = b"agent-evolve:calibrated-portfolio-input-binding:v1\x00"
_PROJECTED_BINDING_DOMAIN = (
    b"agent-evolve:calibrated-portfolio-input-binding:v2\x00"
)
_COMMON_POOL_BINDING_DOMAIN = (
    b"agent-evolve:calibrated-portfolio-input-binding:v4-common-pool-required\x00"
)
_PROPOSAL_SUPPORT_BINDING_DOMAIN = (
    b"agent-evolve:calibrated-portfolio-input-binding:v5-proposal-support\x00"
)
_CONTEXTUAL_ALLOCATION_BINDING_DOMAIN = (
    b"agent-evolve:calibrated-portfolio-input-binding:v6-contextual-allocation\x00"
)
_HIERARCHICAL_POOL_SUPPORT_DOMAIN = (
    b"agent-evolve:calibrated-hierarchical-pool-support:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _canonical_card_keys(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _TOKEN.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of closed card keys")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


def common_pool_required_option_ids(
    request: PortfolioSelectionRequest,
) -> tuple[str, ...]:
    """Return every precommitted action required in the common candidate pool.

    Advisory cards do not alter pool membership.  A bounded dose, however,
    cannot be an honest provider constraint when all of its compatible finite
    actions were removed by an earlier model-blind pool sample.  Binding the
    complete support union into the pool decision closes that failure mode.
    Scientific matched arms may also name the same explicit pool actions in M
    and N, preventing candidate-availability leakage when the neutral view has
    no prompt-visible action evidence.  Both sources remain workload-, model-,
    provider-, and outcome-neutral.
    """

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    hierarchical: list[tuple[str, int]] = []
    for option in request.finite_variation_contract.options:
        metadata = dict(option.metadata)
        if metadata.get(COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY) != (
            CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
        ):
            continue
        try:
            required = int(metadata[COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY])
        except (KeyError, ValueError) as error:
            raise ValueError(
                "hierarchical finite option omitted its pool-support count"
            ) from error
        hierarchical.append((option.option_id, required))
    topology_support: tuple[str, ...] = ()
    if hierarchical:
        counts = {value[1] for value in hierarchical}
        if len(counts) != 1:
            raise ValueError("hierarchical finite options disagree on support count")
        required = next(iter(counts))
        if not 1 <= required < 8 or len(hierarchical) < required:
            raise ValueError("hierarchical finite contract has invalid pool support")
        contract_sha256 = request.finite_variation_contract.identity_sha256
        topology_support = tuple(
            option_id
            for option_id, _count in sorted(
                hierarchical,
                key=lambda value: (
                    hashlib.sha256(
                        _HIERARCHICAL_POOL_SUPPORT_DOMAIN
                        + bytes.fromhex(contract_sha256)
                        + value[0].encode("ascii", errors="strict")
                    ).digest(),
                    value[0],
                ),
            )[:required]
        )
    return tuple(
        sorted(
            {
                *request.candidate_pool_required_option_ids,
                *finite_variation_candidate_pool_required_option_ids(
                    request.finite_variation_contract
                ),
                *topology_support,
                *required_source_evaluation_option_ids(
                    request.finite_variation_contract
                ),
                *(
                    option_id
                    for support in (
                        ()
                        if request.memory_dose_contract is None
                        else request.memory_dose_contract.card_supports
                    )
                    for option_id, _option_identity_sha256 in support.compatible_options
                ),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class CalibratedPortfolioAllocationContext:
    """Prior-only engine facts frozen before one current-wave model call."""

    scope: ForecastCalibrationScope
    wave_index: int
    parent_candidate_identity_sha256: str
    objectives: tuple[SlateMetricObjective, ...]
    assigned_card_keys: tuple[str, ...]
    calibration_snapshot: ForecastCalibrationSnapshot

    def __post_init__(self) -> None:
        if type(self.scope) is not ForecastCalibrationScope:
            raise TypeError("scope must be exact ForecastCalibrationScope")
        self.scope.revalidate()
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        require_sha256(
            self.parent_candidate_identity_sha256,
            "parent_candidate_identity_sha256",
        )
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
            raise ValueError("objectives must use unique canonical metric order")
        _canonical_card_keys(self.assigned_card_keys, name="assigned_card_keys")
        if type(self.calibration_snapshot) is not ForecastCalibrationSnapshot:
            raise TypeError("calibration_snapshot must be exact")
        self.calibration_snapshot.revalidate()
        if self.calibration_snapshot.scope != self.scope:
            raise ValueError("calibration snapshot has a foreign scope")
        if self.calibration_snapshot.cutoff_wave_index_exclusive > self.wave_index:
            raise ValueError(
                "current/future-wave calibration evidence cannot allocate this wave"
            )
        if any(
            value.prediction.wave_index >= self.wave_index
            for value in self.calibration_snapshot.observations
        ):
            raise ValueError(
                "current/future-wave calibration evidence cannot allocate this wave"
            )

    def revalidate(self) -> None:
        if type(self) is not CalibratedPortfolioAllocationContext:
            raise TypeError("context must be exact CalibratedPortfolioAllocationContext")
        CalibratedPortfolioAllocationContext.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "schema_version": 1,
            "scope": self.scope.to_record(),
            "wave_index": self.wave_index,
            "parent_candidate_identity_sha256": (
                self.parent_candidate_identity_sha256
            ),
            "objectives": [value.to_record() for value in self.objectives],
            "assigned_card_keys": list(self.assigned_card_keys),
            "calibration_snapshot": self.calibration_snapshot.to_record(),
        }


@dataclass(frozen=True, slots=True)
class CalibratedPortfolioOptionEvidence:
    """Workload-owned structural projection for one exact sealed option."""

    option_id: str
    option_identity_sha256: str
    locus_key: str
    phenotype_identity_sha256: str
    structural_evidence: SlateStructuralEvidence

    def __post_init__(self) -> None:
        if (
            type(self.option_id) is not str
            or _OPTION_ID.fullmatch(self.option_id) is None
        ):
            raise ValueError("option_id must use the closed token grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.locus_key) is not str or _TOKEN.fullmatch(self.locus_key) is None:
            raise ValueError("locus_key must use the closed token grammar")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        if type(self.structural_evidence) is not SlateStructuralEvidence:
            raise TypeError("structural_evidence must be exact")
        self.structural_evidence.__post_init__()

    def revalidate(self) -> None:
        if type(self) is not CalibratedPortfolioOptionEvidence:
            raise TypeError("evidence must be exact CalibratedPortfolioOptionEvidence")
        CalibratedPortfolioOptionEvidence.__post_init__(self)

    def to_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "locus_key": self.locus_key,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "structural_evidence": self.structural_evidence.to_record(),
        }


def proposal_support_candidates(
    request: PortfolioSelectionRequest,
    option_evidence: tuple[CalibratedPortfolioOptionEvidence, ...],
    common_candidate_pool: CommonCandidatePoolDecision,
) -> tuple[ProposalSupportCandidate, ...]:
    """Project the exact model-visible common pool into structural rows."""

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    if type(option_evidence) is not tuple or any(
        type(value) is not CalibratedPortfolioOptionEvidence
        for value in option_evidence
    ):
        raise TypeError("option_evidence must contain exact evidence rows")
    for value in option_evidence:
        value.revalidate()
    if type(common_candidate_pool) is not CommonCandidatePoolDecision:
        raise TypeError("common_candidate_pool must be an exact decision")
    common_candidate_pool.__post_init__()
    evidence_by_id = {value.option_id: value for value in option_evidence}
    if not set(common_candidate_pool.option_ids).issubset(evidence_by_id):
        raise ValueError("common candidate pool escapes the structural evidence")
    contract = request.finite_variation_contract
    # ``request.__post_init__`` above authenticates the complete finite
    # contract once at this trust boundary.  Resolving every common-pool row
    # through ``contract.resolve`` would revalidate that complete contract for
    # every option, making this projection quadratic for large finite spaces.
    option_by_id = {value.option_id: value for value in contract.options}
    if not set(common_candidate_pool.option_ids).issubset(option_by_id):
        raise ValueError("common candidate pool escapes the sealed finite contract")
    return tuple(
        ProposalSupportCandidate(
            option_id=option_id,
            option_identity_sha256=evidence_by_id[option_id].option_identity_sha256,
            family=option_by_id[option_id].family,
            locus_key=evidence_by_id[option_id].locus_key,
            phenotype_identity_sha256=(
                evidence_by_id[option_id].phenotype_identity_sha256
            ),
            frozen_archive_snapshot_sha256=(
                evidence_by_id[
                    option_id
                ].structural_evidence.frozen_archive_snapshot_sha256
            ),
            structural_evidence_receipt_sha256=(
                evidence_by_id[
                    option_id
                ].structural_evidence.evidence_receipt_sha256
            ),
            archive_novelty_score=(
                evidence_by_id[option_id].structural_evidence.archive_novelty_score
            ),
            structural_coverage_score=(
                evidence_by_id[
                    option_id
                ].structural_evidence.structural_coverage_score
            ),
        )
        for option_id in common_candidate_pool.option_ids
    )


@dataclass(frozen=True, slots=True, eq=False)
class CalibratedPortfolioInputBinding:
    """One request-bound immutable context and all-option evidence snapshot."""

    request_sha256: str
    context: CalibratedPortfolioAllocationContext
    option_evidence: tuple[CalibratedPortfolioOptionEvidence, ...]
    option_prompt_projection: FiniteOptionPromptProjection | None = None
    common_candidate_pool: CommonCandidatePoolDecision | None = None
    proposal_support: ProposalSupportDecision | None = None
    contextual_allocation: ContextualPortfolioAllocationContract | None = None

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.context) is not CalibratedPortfolioAllocationContext:
            raise TypeError("context must be exact CalibratedPortfolioAllocationContext")
        self.context.revalidate()
        if type(self.option_evidence) is not tuple or any(
            type(value) is not CalibratedPortfolioOptionEvidence
            for value in self.option_evidence
        ):
            raise TypeError("option_evidence must contain exact evidence values")
        if not self.option_evidence:
            raise ValueError("option_evidence must be non-empty")
        for value in self.option_evidence:
            value.revalidate()
        option_ids = tuple(value.option_id for value in self.option_evidence)
        if option_ids != tuple(sorted(set(option_ids))):
            raise ValueError("option_evidence must use unique canonical option order")
        archive_snapshots = {
            value.structural_evidence.frozen_archive_snapshot_sha256
            for value in self.option_evidence
        }
        if len(archive_snapshots) != 1:
            raise ValueError("all option evidence must share one frozen archive")
        if self.option_prompt_projection is not None:
            if type(self.option_prompt_projection) is not FiniteOptionPromptProjection:
                raise TypeError(
                    "option_prompt_projection must be exact prompt projection"
                )
            self.option_prompt_projection.__post_init__()
        if self.common_candidate_pool is not None:
            if type(self.common_candidate_pool) is not CommonCandidatePoolDecision:
                raise TypeError(
                    "common_candidate_pool must be an exact decision or None"
                )
            self.common_candidate_pool.__post_init__()
        if self.proposal_support is not None:
            if type(self.proposal_support) is not ProposalSupportDecision:
                raise TypeError(
                    "proposal_support must be an exact decision or None"
                )
            self.proposal_support.__post_init__()
            if self.common_candidate_pool is None:
                raise ValueError(
                    "proposal support requires a task-keyed common candidate pool"
                )
        if self.contextual_allocation is not None:
            if type(self.contextual_allocation) is not (
                ContextualPortfolioAllocationContract
            ):
                raise TypeError(
                    "contextual_allocation must be an exact contract or None"
                )
            self.contextual_allocation.__post_init__()
            if (
                self.contextual_allocation.campaign_generation
                != self.context.wave_index
            ):
                raise ValueError(
                    "contextual allocation differs from the campaign generation"
                )

    def revalidate(self) -> None:
        if type(self) is not CalibratedPortfolioInputBinding:
            raise TypeError("binding must be exact CalibratedPortfolioInputBinding")
        CalibratedPortfolioInputBinding.__post_init__(self)

    def require_request(self, request: PortfolioSelectionRequest) -> None:
        """Authenticate the complete binding against one exact public request."""

        self.revalidate()
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be exact PortfolioSelectionRequest")
        request.__post_init__()
        if self.request_sha256 != request.request_sha256:
            raise ValueError("calibrated input binding names a foreign request")
        contract = request.finite_variation_contract
        if (
            self.context.parent_candidate_identity_sha256
            != contract.parent_configuration_sha256
        ):
            raise ValueError("allocation context names a foreign parent")
        if tuple(value.metric_id for value in self.context.objectives) != (
            request.required_metric_ids
        ):
            raise ValueError("allocation objectives differ from requested metrics")
        request_card_keys = {value.card_key for value in request.cards}
        if not set(self.context.assigned_card_keys).issubset(request_card_keys):
            raise ValueError("assigned cards escape the request snapshot")
        expected = {
            value.option_id: value.identity_sha256 for value in contract.options
        }
        observed = {
            value.option_id: value.option_identity_sha256
            for value in self.option_evidence
        }
        if observed != expected:
            raise ValueError("option evidence differs from the sealed finite contract")
        if self.option_prompt_projection is not None:
            self.option_prompt_projection.require_contract(contract)
        if self.common_candidate_pool is not None:
            policy = TaskKeyedCommonCandidatePoolPolicy(
                replicate_seed=self.common_candidate_pool.replicate_seed,
                candidate_pool_size=(
                    self.common_candidate_pool.candidate_pool_size
                ),
                model_selection_size=(
                    self.common_candidate_pool.model_selection_size
                ),
            )
            policy.require_decision(
                self.common_candidate_pool,
                benchmark_sha256=self.context.scope.benchmark_sha256,
                wave_index=self.context.wave_index,
                parent_configuration_sha256=(
                    self.context.parent_candidate_identity_sha256
                ),
                contract=contract,
                evaluation_size=request.portfolio_size,
                min_distinct_families=request.min_distinct_families,
                require_pairwise_disjoint_parent_patches=(
                    request.require_pairwise_disjoint_parent_patches
                ),
                required_option_ids=common_pool_required_option_ids(request),
                certified_feasibility_witness_option_ids=(
                    self.common_candidate_pool.certified_feasibility_witness_option_ids
                ),
            )
        if self.proposal_support is not None:
            assert self.common_candidate_pool is not None
            candidates = proposal_support_candidates(
                request,
                self.option_evidence,
                self.common_candidate_pool,
            )
            expected_support = StructuralProposalSupportPolicy().select(
                request_sha256=request.request_sha256,
                common_candidate_pool_decision_sha256=(
                    self.common_candidate_pool.decision_sha256
                ),
                model_selection_size=(
                    self.common_candidate_pool.model_selection_size
                ),
                candidates=candidates,
            )
            if self.proposal_support != expected_support:
                raise ValueError(
                    "proposal-support decision differs from the sealed input binding"
                )
        if self.contextual_allocation is not None:
            if self.contextual_allocation.evaluation_slots != request.portfolio_size:
                raise ValueError(
                    "contextual allocation differs from the evaluation width"
                )

    def prompt_records_for(
        self,
        request: PortfolioSelectionRequest,
    ) -> tuple[dict[str, object], ...]:
        """Return only this binding's authenticated model-visible option view."""

        self.require_request(request)
        records = (
            request.finite_variation_contract.prompt_records()
            if self.option_prompt_projection is None
            else self.option_prompt_projection.prompt_records()
        )
        if self.common_candidate_pool is None:
            return records
        by_id = {str(value["option_id"]): value for value in records}
        return tuple(
            by_id[option_id]
            for option_id in self.common_candidate_pool.option_ids
        )

    def prompt_projection_contract_for(
        self,
        request: PortfolioSelectionRequest,
    ) -> dict[str, object] | None:
        """Return the compact projection receipt, if this binding opted in."""

        self.require_request(request)
        if self.option_prompt_projection is None:
            return None
        return self.option_prompt_projection.to_prompt_contract_record()

    def evidence_for(
        self,
        request: PortfolioSelectionRequest,
        option: FiniteVariationOption,
    ) -> CalibratedPortfolioOptionEvidence:
        self.require_request(request)
        if type(option) is not FiniteVariationOption:
            raise TypeError("option must be exact FiniteVariationOption")
        resolved = request.finite_variation_contract.resolve(option.option_id)
        if resolved != option:
            raise ValueError("option belongs to a foreign finite contract")
        evidence = next(
            value for value in self.option_evidence if value.option_id == option.option_id
        )
        if evidence.option_identity_sha256 != option.identity_sha256:
            raise ValueError("structural evidence belongs to a foreign option")
        return evidence

    def _unsigned_record(self) -> dict[str, object]:
        self.revalidate()
        record: dict[str, object] = {
            "schema_version": (
                6
                if self.contextual_allocation is not None
                else 5
                if self.proposal_support is not None
                else 4
                if self.common_candidate_pool is not None
                else (1 if self.option_prompt_projection is None else 2)
            ),
            "request_sha256": self.request_sha256,
            "context": self.context.to_record(),
            "option_evidence": [value.to_record() for value in self.option_evidence],
        }
        # Absence is the legacy representation.  Do not add a null field: that
        # would silently invalidate every finalized v2/four-role binding.
        if self.option_prompt_projection is not None:
            record["option_prompt_projection"] = (
                self.option_prompt_projection.to_binding_record()
            )
        if self.common_candidate_pool is not None:
            record["common_candidate_pool"] = self.common_candidate_pool.to_record()
        if self.proposal_support is not None:
            record["proposal_support"] = self.proposal_support.to_record()
        if self.contextual_allocation is not None:
            record["contextual_allocation"] = (
                self.contextual_allocation.to_record()
            )
        return record

    @property
    def binding_sha256(self) -> str:
        domain = (
            _CONTEXTUAL_ALLOCATION_BINDING_DOMAIN
            if self.contextual_allocation is not None
            else _PROPOSAL_SUPPORT_BINDING_DOMAIN
            if self.proposal_support is not None
            else _COMMON_POOL_BINDING_DOMAIN
            if self.common_candidate_pool is not None
            else (
                _BINDING_DOMAIN
                if self.option_prompt_projection is None
                else _PROJECTED_BINDING_DOMAIN
            )
        )
        return hashlib.sha256(
            domain + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is CalibratedPortfolioInputBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


CalibratedPortfolioBindingProvider = Callable[
    [PortfolioSelectionRequest], CalibratedPortfolioInputBinding
]


__all__ = [
    "CalibratedPortfolioAllocationContext",
    "CalibratedPortfolioBindingProvider",
    "CalibratedPortfolioInputBinding",
    "CalibratedPortfolioOptionEvidence",
    "common_pool_required_option_ids",
    "proposal_support_candidates",
]
