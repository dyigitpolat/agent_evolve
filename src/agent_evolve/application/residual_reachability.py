"""Workload-neutral parent coverage and on-demand residual composition.

Pareto archives summarize objective quality, but they need not preserve useful
genotype basins.  This module maintains a bounded reachability basis over
already evaluated candidates and exposes a compact cross-parent finite-action
schema.  Proposal experts select only a parent ID and opaque atomic option IDs;
trusted engine code resolves and materializes the exact child.

The module deliberately contains no workload, objective, model, or provider
identity.  It is downstream of a benchmark's :class:`FiniteVariationCatalog`
and upstream of the real-evaluation broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchRecombiner,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_BASIS_POLICY_DOMAIN = b"agent-evolve:residual-reachability-basis-policy:v1\x00"
_BASIS_DOMAIN = b"agent-evolve:residual-reachability-basis:v1\x00"
_SCHEMA_DOMAIN = b"agent-evolve:cross-parent-finite-action-schema:v1\x00"
_PLAN_DOMAIN = b"agent-evolve:hierarchical-residual-plan:v1\x00"
_MATERIALIZATION_DOMAIN = b"agent-evolve:residual-plan-materialization:v1\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _token(value: str, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _candidate(value: CandidateId, name: str = "candidate_id") -> None:
    if type(value) is not CandidateId:
        raise TypeError(f"{name} must be an exact CandidateId")
    CandidateId.__post_init__(value)


class ResidualProposalRole(str, Enum):
    """Scientific purpose of a proposal, independent of its workload."""

    LOCAL_EXPLOIT = "local_exploit"
    INTERACTION = "interaction"
    DONOR_RECOMBINATION = "donor_recombination"
    STRUCTURAL_COVERAGE = "structural_coverage"
    RESTART = "restart"


class ReachabilityAdmissionReason(str, Enum):
    """Authenticated route by which an evaluated parent enters the basis."""

    QUALITY_ARCHIVE = "quality_archive"
    INITIAL_DESIGN = "initial_design"
    EARNED_LINEAGE = "earned_lineage"
    STRUCTURAL_COVER = "structural_cover"
    CAPACITY_FILL = "capacity_fill"


@dataclass(frozen=True, slots=True)
class ReachabilityCandidate:
    """Prompt-independent evidence for one already evaluated parent."""

    candidate_id: CandidateId
    configuration: FrozenJsonObject
    phenotype_identity_sha256: str
    evaluation_ordinal: int
    structural_cell: str
    quality_archive_member: bool
    initial_design_member: bool
    earned_positive_lineage: bool

    def __post_init__(self) -> None:
        _candidate(self.candidate_id)
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        if freeze_json(self.configuration) is not self.configuration:
            raise TypeError("configuration must already be frozen typed JSON")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        if type(self.evaluation_ordinal) is not int or self.evaluation_ordinal <= 0:
            raise ValueError("evaluation_ordinal must be positive")
        _token(self.structural_cell, "structural_cell")
        for name in (
            "quality_archive_member",
            "initial_design_member",
            "earned_positive_lineage",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return typed_json_sha256(self.configuration)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate_id": self.candidate_id.value,
            "configuration_sha256": self.configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "evaluation_ordinal": self.evaluation_ordinal,
            "structural_cell": self.structural_cell,
            "quality_archive_member": self.quality_archive_member,
            "initial_design_member": self.initial_design_member,
            "earned_positive_lineage": self.earned_positive_lineage,
        }


@dataclass(frozen=True, slots=True)
class ResidualReachabilityBasisPolicy:
    """Bounded, deterministic dual-archive parent retention policy."""

    maximum_parents: int = 32
    maximum_quality_archive_parents: int = 16
    maximum_initial_design_parents: int = 16
    maximum_earned_lineage_parents: int = 8
    maximum_structural_cover_parents: int = 8
    policy_id: str = "bounded_dual_archive_reachability_basis"
    policy_version: int = 1

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        for name in (
            "maximum_parents",
            "maximum_quality_archive_parents",
            "maximum_initial_design_parents",
            "maximum_earned_lineage_parents",
            "maximum_structural_cover_parents",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
            if name != "maximum_parents" and value > self.maximum_parents:
                raise ValueError(f"{name} cannot exceed maximum_parents")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "maximum_parents": self.maximum_parents,
            "maximum_quality_archive_parents": (
                self.maximum_quality_archive_parents
            ),
            "maximum_initial_design_parents": self.maximum_initial_design_parents,
            "maximum_earned_lineage_parents": self.maximum_earned_lineage_parents,
            "maximum_structural_cover_parents": (
                self.maximum_structural_cover_parents
            ),
            "outcomes_consulted": [
                "quality_archive_membership",
                "earned_positive_lineage",
            ],
            "forbidden_fields": ["workload_id", "model_id", "provider_id"],
        }

    @property
    def definition_sha256(self) -> str:
        return _hash(_BASIS_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "definition_sha256": self.definition_sha256}


@dataclass(frozen=True, slots=True)
class ReachabilityBasisMember:
    candidate: ReachabilityCandidate
    admission_reasons: tuple[ReachabilityAdmissionReason, ...]

    def __post_init__(self) -> None:
        if type(self.candidate) is not ReachabilityCandidate:
            raise TypeError("candidate must be an exact ReachabilityCandidate")
        ReachabilityCandidate.__post_init__(self.candidate)
        if type(self.admission_reasons) is not tuple or not self.admission_reasons:
            raise ValueError("admission_reasons must be a non-empty exact tuple")
        if any(type(value) is not ReachabilityAdmissionReason for value in self.admission_reasons):
            raise TypeError("admission_reasons contain a foreign value")
        canonical = tuple(sorted(set(self.admission_reasons), key=lambda value: value.value))
        if self.admission_reasons != canonical:
            raise ValueError("admission_reasons must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate": self.candidate.to_record(),
            "admission_reasons": [value.value for value in self.admission_reasons],
        }


@dataclass(frozen=True, slots=True)
class ResidualReachabilityBasis:
    policy_definition_sha256: str
    source_candidate_count: int
    members: tuple[ReachabilityBasisMember, ...]
    basis_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.source_candidate_count) is not int or self.source_candidate_count <= 0:
            raise ValueError("source_candidate_count must be positive")
        if type(self.members) is not tuple or not self.members:
            raise ValueError("members must be a non-empty exact tuple")
        if any(type(value) is not ReachabilityBasisMember for value in self.members):
            raise TypeError("members contain a foreign value")
        for value in self.members:
            ReachabilityBasisMember.__post_init__(value)
        ids = tuple(value.candidate.candidate_id.value for value in self.members)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("members must have unique canonical candidate order")
        if len(self.members) > self.source_candidate_count:
            raise ValueError("basis cannot exceed its source population")
        object.__setattr__(self, "basis_sha256", _hash(_BASIS_DOMAIN, self._unsigned_record()))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_definition_sha256": self.policy_definition_sha256,
            "source_candidate_count": self.source_candidate_count,
            "members": [value.to_record() for value in self.members],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "basis_sha256": self.basis_sha256}


def select_residual_reachability_basis(
    candidates: tuple[ReachabilityCandidate, ...],
    policy: ResidualReachabilityBasisPolicy,
) -> ResidualReachabilityBasis:
    """Retain quality and bounded structural stepping-stone parents.

    Selection is deterministic.  Earlier evaluated candidates break ties so a
    later outcome cannot retroactively change a frozen prefix basis.
    """

    if type(candidates) is not tuple or not candidates:
        raise ValueError("candidates must be a non-empty exact tuple")
    if any(type(value) is not ReachabilityCandidate for value in candidates):
        raise TypeError("candidates contain a foreign value")
    for value in candidates:
        ReachabilityCandidate.__post_init__(value)
    if type(policy) is not ResidualReachabilityBasisPolicy:
        raise TypeError("policy must be an exact ResidualReachabilityBasisPolicy")
    policy.__post_init__()
    ids = tuple(value.candidate_id for value in candidates)
    phenotypes = tuple(value.phenotype_identity_sha256 for value in candidates)
    ordinals = tuple(value.evaluation_ordinal for value in candidates)
    if len(set(ids)) != len(ids) or len(set(phenotypes)) != len(phenotypes):
        raise ValueError("source candidates must have unique IDs and phenotypes")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("source candidates must have unique evaluation ordinals")

    ordered = tuple(sorted(candidates, key=lambda value: value.evaluation_ordinal))
    reasons: dict[CandidateId, set[ReachabilityAdmissionReason]] = {}
    selected: dict[CandidateId, ReachabilityCandidate] = {}

    def admit(
        pool: tuple[ReachabilityCandidate, ...],
        reason: ReachabilityAdmissionReason,
        limit: int,
    ) -> None:
        admitted = 0
        for value in pool:
            if admitted >= limit or len(selected) >= policy.maximum_parents:
                break
            if value.candidate_id not in selected:
                selected[value.candidate_id] = value
                admitted += 1
            reasons.setdefault(value.candidate_id, set()).add(reason)

    admit(
        tuple(value for value in ordered if value.quality_archive_member),
        ReachabilityAdmissionReason.QUALITY_ARCHIVE,
        policy.maximum_quality_archive_parents,
    )
    admit(
        tuple(value for value in ordered if value.initial_design_member),
        ReachabilityAdmissionReason.INITIAL_DESIGN,
        policy.maximum_initial_design_parents,
    )
    admit(
        tuple(value for value in ordered if value.earned_positive_lineage),
        ReachabilityAdmissionReason.EARNED_LINEAGE,
        policy.maximum_earned_lineage_parents,
    )

    covered_cells = {value.structural_cell for value in selected.values()}
    structural_pool: list[ReachabilityCandidate] = []
    for value in ordered:
        if value.candidate_id in selected or value.structural_cell in covered_cells:
            continue
        structural_pool.append(value)
        covered_cells.add(value.structural_cell)
    admit(
        tuple(structural_pool),
        ReachabilityAdmissionReason.STRUCTURAL_COVER,
        policy.maximum_structural_cover_parents,
    )

    if len(selected) < policy.maximum_parents:
        admit(
            tuple(value for value in reversed(ordered) if value.candidate_id not in selected),
            ReachabilityAdmissionReason.CAPACITY_FILL,
            policy.maximum_parents - len(selected),
        )

    members = tuple(
        ReachabilityBasisMember(
            candidate=value,
            admission_reasons=tuple(sorted(reasons[value.candidate_id], key=lambda item: item.value)),
        )
        for value in sorted(selected.values(), key=lambda item: item.candidate_id.value)
    )
    return ResidualReachabilityBasis(
        policy_definition_sha256=policy.definition_sha256,
        source_candidate_count=len(candidates),
        members=members,
    )


@dataclass(frozen=True, slots=True)
class ParentFiniteVariationBinding:
    parent_candidate_id: CandidateId
    contract: FiniteVariationContract

    def __post_init__(self) -> None:
        _candidate(self.parent_candidate_id, "parent_candidate_id")
        if type(self.contract) is not FiniteVariationContract:
            raise TypeError("contract must be an exact FiniteVariationContract")
        validate_finite_variation_contract(self.contract)


@dataclass(frozen=True, slots=True)
class CrossParentFiniteActionSchema:
    """Prompt-compact union of compatible parent-bound finite contracts."""

    bindings: tuple[ParentFiniteVariationBinding, ...]
    action_prompt_records: tuple[dict[str, object], ...]
    schema_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.bindings) is not tuple or not self.bindings:
            raise ValueError("bindings must be a non-empty exact tuple")
        if any(type(value) is not ParentFiniteVariationBinding for value in self.bindings):
            raise TypeError("bindings contain a foreign value")
        for value in self.bindings:
            ParentFiniteVariationBinding.__post_init__(value)
        ids = tuple(value.parent_candidate_id.value for value in self.bindings)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("bindings must use unique canonical parent order")
        if type(self.action_prompt_records) is not tuple or not self.action_prompt_records:
            raise ValueError("action_prompt_records must be non-empty")
        option_ids = tuple(str(value.get("option_id")) for value in self.action_prompt_records)
        if option_ids != tuple(sorted(set(option_ids))):
            raise ValueError("action prompt records must use canonical option order")
        object.__setattr__(self, "schema_sha256", _hash(_SCHEMA_DOMAIN, self._unsigned_record()))

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(str(value["option_id"]) for value in self.action_prompt_records)

    def contract_for(self, parent_id: CandidateId) -> FiniteVariationContract:
        _candidate(parent_id, "parent_id")
        matches = tuple(value.contract for value in self.bindings if value.parent_candidate_id == parent_id)
        if len(matches) != 1:
            raise ValueError("parent_id is outside the cross-parent schema")
        return matches[0]

    def _unsigned_record(self) -> dict[str, object]:
        union_ids = set(self.option_ids)
        return {
            "schema_version": 1,
            "action_prompt_records": list(self.action_prompt_records),
            "parents": [
                {
                    "parent_candidate_id": value.parent_candidate_id.value,
                    "contract_identity_sha256": value.contract.identity_sha256,
                    "unavailable_option_ids": sorted(
                        union_ids - {option.option_id for option in value.contract.options}
                    ),
                }
                for value in self.bindings
            ],
            "model_materialization_authority": False,
        }

    def prompt_record(self) -> dict[str, object]:
        """Return compact semantic choices; retain hashes in the schema receipt.

        Catalog, action-definition, and source digests are valuable audit
        evidence but waste model attention when repeated across hundreds of
        choices.  The schema digest already authenticates those fields.  The
        prompt projection therefore removes metadata keys ending in
        ``_sha256`` while preserving action IDs, families, descriptions, and
        non-digest semantics.
        """

        self.__post_init__()
        union_ids = set(self.option_ids)
        return {
            "schema_version": 1,
            "schema_sha256": self.schema_sha256,
            "actions": [
                {
                    "option_id": value["option_id"],
                    "family": value["family"],
                    "description": value["description"],
                    "metadata": {
                        key: item
                        for key, item in dict(value["metadata"]).items()
                        if not key.endswith("_sha256")
                    },
                    "parent_specific_description_omitted": value[
                        "parent_specific_description_omitted"
                    ],
                }
                for value in self.action_prompt_records
            ],
            "parents": [
                {
                    "parent_candidate_id": value.parent_candidate_id.value,
                    "unavailable_option_ids": sorted(
                        union_ids
                        - {option.option_id for option in value.contract.options}
                    ),
                }
                for value in self.bindings
            ],
            "model_selects_parent_and_atomic_ids_only": True,
            "model_materialization_authority": False,
        }

    def evidence_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "schema_sha256": self.schema_sha256}


def bind_cross_parent_finite_action_schema(
    bindings: tuple[ParentFiniteVariationBinding, ...],
) -> CrossParentFiniteActionSchema:
    """Deduplicate action semantics while retaining exact parent eligibility.

    A catalog description may name the current parent value (for example,
    ``replace X with Y``), so exact description text need not be stable across
    parents.  Family and structured semantic metadata remain authoritative.
    Metadata values whose keys end in ``_sha256`` are parent-bound evidence
    (for example, a compiled-child phenotype digest), not shared action
    meaning.  Every exact value remains authenticated by its parent contract
    identity, while the common schema records only those evidence-key names.
    When semantic fields agree but descriptions differ, the common schema
    records that the parent-specific prose was omitted; proposal context can
    still expose the exact parent configuration.  Any disagreement in family
    or non-digest metadata fails closed.
    """

    if type(bindings) is not tuple or not bindings:
        raise ValueError("bindings must be a non-empty exact tuple")
    canonical = tuple(sorted(bindings, key=lambda value: value.parent_candidate_id.value))
    if bindings != canonical:
        raise ValueError("bindings must already use canonical parent order")
    semantics_by_id: dict[str, tuple[str, dict[str, object]]] = {}
    descriptions_by_id: dict[str, set[str]] = {}
    evidence_keys_by_id: dict[str, set[str]] = {}
    for binding in bindings:
        ParentFiniteVariationBinding.__post_init__(binding)
        for option in binding.contract.options:
            prompt = option.prompt_record()
            metadata = dict(prompt["metadata"])
            semantics = (
                option.family,
                {
                    key: item
                    for key, item in metadata.items()
                    if not key.endswith("_sha256")
                },
            )
            previous = semantics_by_id.setdefault(option.option_id, semantics)
            if previous != semantics:
                raise ValueError(
                    "an option ID has inconsistent prompt semantics across parents"
                )
            descriptions_by_id.setdefault(option.option_id, set()).add(
                option.description
            )
            evidence_keys_by_id.setdefault(option.option_id, set()).update(
                key for key in metadata if key.endswith("_sha256")
            )
    prompt_by_id: dict[str, dict[str, object]] = {}
    for option_id in sorted(semantics_by_id):
        family, metadata = semantics_by_id[option_id]
        descriptions = descriptions_by_id[option_id]
        prompt_by_id[option_id] = {
            "option_id": option_id,
            "family": family,
            "description": next(iter(descriptions)) if len(descriptions) == 1 else None,
            "metadata": metadata,
            "parent_bound_evidence_metadata_keys": sorted(
                evidence_keys_by_id[option_id]
            ),
            "parent_specific_description_omitted": len(descriptions) != 1,
        }
    return CrossParentFiniteActionSchema(
        bindings=bindings,
        action_prompt_records=tuple(prompt_by_id[key] for key in sorted(prompt_by_id)),
    )


@dataclass(frozen=True, slots=True)
class HierarchicalResidualPlan:
    """One model- or engine-proposed parent plus atomic-ID tuple."""

    parent_candidate_id: CandidateId
    parent_contract_sha256: str
    action_schema_sha256: str
    component_option_ids: tuple[str, ...]
    role: ResidualProposalRole
    expert_id: str
    expert_definition_sha256: str
    native_rank: int
    decision_receipt_sha256: str

    def __post_init__(self) -> None:
        _candidate(self.parent_candidate_id, "parent_candidate_id")
        for name in (
            "parent_contract_sha256",
            "action_schema_sha256",
            "expert_definition_sha256",
            "decision_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.component_option_ids) is not tuple
            or not 1 <= len(self.component_option_ids) <= 2
        ):
            raise ValueError("component_option_ids must have radius one or two")
        if any(type(value) is not str or not value for value in self.component_option_ids):
            raise TypeError("component_option_ids must contain non-empty strings")
        if self.component_option_ids != tuple(sorted(set(self.component_option_ids))):
            raise ValueError("component_option_ids must be unique and canonical")
        if type(self.role) is not ResidualProposalRole:
            raise TypeError("role must be an exact ResidualProposalRole")
        _token(self.expert_id, "expert_id")
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be positive")

    @property
    def radius(self) -> int:
        return len(self.component_option_ids)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_contract_sha256": self.parent_contract_sha256,
            "action_schema_sha256": self.action_schema_sha256,
            "component_option_ids": list(self.component_option_ids),
            "radius": self.radius,
            "role": self.role.value,
            "expert_id": self.expert_id,
            "expert_definition_sha256": self.expert_definition_sha256,
            "native_rank": self.native_rank,
            "decision_receipt_sha256": self.decision_receipt_sha256,
            "model_authored_configuration": False,
        }

    @property
    def plan_sha256(self) -> str:
        return _hash(_PLAN_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class MaterializedResidualProposal:
    plan: HierarchicalResidualPlan
    target_candidate_id: CandidateId
    configuration: FrozenJsonObject
    component_option_identity_sha256s: tuple[str, ...]
    engine_materialization_receipt_sha256: str
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not HierarchicalResidualPlan:
            raise TypeError("plan must be an exact HierarchicalResidualPlan")
        HierarchicalResidualPlan.__post_init__(self.plan)
        _candidate(self.target_candidate_id, "target_candidate_id")
        if self.target_candidate_id == self.plan.parent_candidate_id:
            raise ValueError("target candidate must differ from its parent")
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        if freeze_json(self.configuration) is not self.configuration:
            raise TypeError("configuration must already be frozen typed JSON")
        if (
            type(self.component_option_identity_sha256s) is not tuple
            or len(self.component_option_identity_sha256s) != self.plan.radius
        ):
            raise ValueError("component identities must match the plan radius")
        for value in self.component_option_identity_sha256s:
            require_sha256(value, "component_option_identity_sha256")
        require_sha256(
            self.engine_materialization_receipt_sha256,
            "engine_materialization_receipt_sha256",
        )
        object.__setattr__(self, "proposal_sha256", _hash(_MATERIALIZATION_DOMAIN, self._unsigned_record()))

    @property
    def configuration_sha256(self) -> str:
        return typed_json_sha256(self.configuration)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_sha256": self.plan.plan_sha256,
            "target_candidate_id": self.target_candidate_id.value,
            "configuration_sha256": self.configuration_sha256,
            "component_option_identity_sha256s": list(
                self.component_option_identity_sha256s
            ),
            "engine_materialization_receipt_sha256": (
                self.engine_materialization_receipt_sha256
            ),
        }

    def to_record(self, *, include_configuration: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {**self._unsigned_record(), "proposal_sha256": self.proposal_sha256}
        if include_configuration:
            record["configuration"] = self.configuration
        return record


def materialize_hierarchical_residual_plan(
    *,
    schema: CrossParentFiniteActionSchema,
    plan: HierarchicalResidualPlan,
    target_candidate_id: CandidateId,
) -> MaterializedResidualProposal:
    """Resolve and replay a radius-one or safe radius-two residual plan."""

    if type(schema) is not CrossParentFiniteActionSchema:
        raise TypeError("schema must be an exact CrossParentFiniteActionSchema")
    schema.__post_init__()
    if type(plan) is not HierarchicalResidualPlan:
        raise TypeError("plan must be an exact HierarchicalResidualPlan")
    plan.__post_init__()
    _candidate(target_candidate_id, "target_candidate_id")
    if plan.action_schema_sha256 != schema.schema_sha256:
        raise ValueError("plan is bound to a different cross-parent schema")
    contract = schema.contract_for(plan.parent_candidate_id)
    if plan.parent_contract_sha256 != contract.identity_sha256:
        raise ValueError("plan is bound to a different parent contract")
    options = tuple(contract.resolve(value) for value in plan.component_option_ids)
    if len(options) == 1:
        configuration = options[0].child_configuration
        receipt = _hash(
            _MATERIALIZATION_DOMAIN,
            {
                "kind": "sealed_atomic_option",
                "plan_sha256": plan.plan_sha256,
                "contract_identity_sha256": contract.identity_sha256,
                "option_identity_sha256": options[0].identity_sha256,
                "configuration_sha256": options[0].child_configuration_sha256,
            },
        )
    else:
        materialization = DisjointPatchRecombiner().materialize(
            ancestor=contract.parent_configuration,
            ancestor_candidate_id=plan.parent_candidate_id,
            left=options[0].child_configuration,
            left_candidate_id=CandidateId(
                f"candidate_residual_component_left_{plan.plan_sha256[:16]}"
            ),
            right=options[1].child_configuration,
            right_candidate_id=CandidateId(
                f"candidate_residual_component_right_{plan.plan_sha256[:16]}"
            ),
            target_candidate_id=target_candidate_id,
        )
        if type(materialization.configuration) is not FrozenJsonObject:
            raise TypeError("residual materialization must retain an object root")
        configuration = materialization.configuration
        receipt = materialization.receipt_sha256
    return MaterializedResidualProposal(
        plan=plan,
        target_candidate_id=target_candidate_id,
        configuration=configuration,
        component_option_identity_sha256s=tuple(
            value.identity_sha256 for value in options
        ),
        engine_materialization_receipt_sha256=receipt,
    )


__all__ = [
    "CrossParentFiniteActionSchema",
    "HierarchicalResidualPlan",
    "MaterializedResidualProposal",
    "ParentFiniteVariationBinding",
    "ReachabilityAdmissionReason",
    "ReachabilityBasisMember",
    "ReachabilityCandidate",
    "ResidualProposalRole",
    "ResidualReachabilityBasis",
    "ResidualReachabilityBasisPolicy",
    "bind_cross_parent_finite_action_schema",
    "materialize_hierarchical_residual_plan",
    "select_residual_reachability_basis",
]
