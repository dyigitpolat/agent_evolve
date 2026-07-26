"""Generic three-generation causal-development screen for AgentEvolve.

This module owns no benchmark semantics.  A narrow boundary supplies frozen
parent-relative finite catalogs and authenticated hypothesis compilation.  The
planner composes existing causal-memory, strict-treatment, deterministic
materialization, recombination, and budgeted-optimizer mechanisms into the
preregistered G1/G2/G3 screen.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InsightAssignmentKind,
    InvocationPlan,
    InvocationOutcome,
    MaterializedInvocation,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    FrozenWaveReward,
    GenerationPlan,
    GenerationReceipt,
    OptimizerBudget,
    OptimizerSlot,
    OptimizerState,
)
from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
    context_stratum_hash,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
)
from agent_evolve.application.staged_memory import (
    DiagnosticMemoryCheckpointService,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
    require_sha256,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    freeze_json,
    is_frozen_json_value,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.memory.prompt_shape import (
    MatchedPromptStructureReceipt,
    seal_matched_prompt_structure,
)
from agent_evolve.policies.memory.staged_causal import (
    CausalSearchScorePolicy,
    DeterministicMemoryControlPolicy,
    FrozenDiagnosticMemoryWave,
    MemoryAssignmentArm,
    MemoryCheckpointClosure,
    MemoryCheckpointClosureStatus,
    ResolvedInsightAssignment,
    WaveSealedCheckpointBuilder,
)
from agent_evolve.policies.memory.treatment_compliance import (
    InsightTreatmentRequirement,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentInsightEvidence,
)
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchMaterialization,
    DisjointPatchRecombiner,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
    SourceAttribution,
    CandidateDraft,
)
from agent_evolve.ports.executable_hypothesis import (
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    validate_hypothesis_compilation,
)
from agent_evolve.ports.id_factory import IdFactory


G3_SCREEN_POLICY_ID = "g3_causal_development_screen"
G3_SCREEN_POLICY_VERSION = 1
G3_SCREEN_BUDGET = OptimizerBudget(
    max_unique_evaluations=11,
    max_logical_llm_calls=6,
    max_generations=3,
)
G1_DIAGNOSTIC_SLOT_IDS = ("g1_diagnostic_0", "g1_diagnostic_1")
G2_SLOT_IDS = ("g2_adaptive", "g2_score_shuffled", "g2_sham", "g2_mate")
G3_SLOT_IDS = (
    "g3_reproduction",
    "g3_adaptive_union",
    "g3_score_shuffled_union",
    "g3_sham_union",
)

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_CHOICE_DOMAIN = b"agent-evolve:g3-parent-bound-action-choice:v1\x00"
_PERMUTATION_DOMAIN = b"agent-evolve:g3-diagnostic-joint-permutation:v1\x00"
_OCCURRENCE_DOMAIN = b"agent-evolve:g3-seed-occurrence-binding:v1\x00"
_PROSPECTIVE_DOMAIN = b"agent-evolve:g3-prospective-endpoint-proof:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the union.
            raise AssertionError("unsupported path segment")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class ParentBoundActionChoice:
    """Outcome-blind exact option commitment made during preparation."""

    role: str
    catalog_id: str
    parent_configuration_sha256: str
    finite_contract_sha256: str
    option_id: str
    option_identity_sha256: str
    selection_policy_id: str
    selection_policy_version: int
    selection_policy_definition_sha256: str
    choice_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("role", "catalog_id", "selection_policy_id"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError(f"{name} must use the canonical token grammar")
        for name in (
            "parent_configuration_sha256",
            "finite_contract_sha256",
            "option_identity_sha256",
            "selection_policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be canonical non-empty text")
        if (
            type(self.selection_policy_version) is not int
            or self.selection_policy_version <= 0
        ):
            raise ValueError("selection_policy_version must be positive")
        object.__setattr__(self, "choice_sha256", _hash(_CHOICE_DOMAIN, self.to_record()))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "role": self.role,
            "catalog_id": self.catalog_id,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "finite_contract_sha256": self.finite_contract_sha256,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "selection_policy_id": self.selection_policy_id,
            "selection_policy_version": self.selection_policy_version,
            "selection_policy_definition_sha256": (
                self.selection_policy_definition_sha256
            ),
        }

    @classmethod
    def seal(
        cls,
        *,
        role: str,
        contract: FiniteVariationContract,
        option_id: str,
        selection_policy_id: str,
        selection_policy_version: int,
        selection_policy_definition_sha256: str,
    ) -> "ParentBoundActionChoice":
        validate_finite_variation_contract(contract)
        option = contract.resolve(option_id)
        return cls(
            role=role,
            catalog_id=contract.catalog_id,
            parent_configuration_sha256=contract.parent_configuration_sha256,
            finite_contract_sha256=contract.identity_sha256,
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            selection_policy_id=selection_policy_id,
            selection_policy_version=selection_policy_version,
            selection_policy_definition_sha256=selection_policy_definition_sha256,
        )

    def validate_contract(self, contract: FiniteVariationContract) -> None:
        validate_finite_variation_contract(contract)
        observed = (
            contract.catalog_id,
            contract.parent_configuration_sha256,
            contract.identity_sha256,
        )
        expected = (
            self.catalog_id,
            self.parent_configuration_sha256,
            self.finite_contract_sha256,
        )
        if observed != expected:
            raise ValueError("parent-bound action choice differs from finite contract")
        option = contract.resolve(self.option_id)
        if option.identity_sha256 != self.option_identity_sha256:
            raise ValueError("parent-bound action option identity changed")


@dataclass(frozen=True, slots=True)
class FrozenDiagnosticPermutation:
    """Public randomization realization for the complete two-slot G1 block.

    Randomness lives outside the planner.  Preparation samples one integer
    uniformly from ``[0, 2!)`` and records the sampler identity here; the
    planner performs only the deterministic rank-to-joint-assignment mapping.
    This is a joint receipt, not two independent probability annotations.
    """

    active_references: tuple[InsightRef, InsightRef]
    permutation_rank: int
    randomization_policy_id: str
    randomization_policy_version: int
    randomization_definition_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.active_references) is not tuple
            or len(self.active_references) != 2
            or self.active_references
            != tuple(sorted(set(self.active_references)))
        ):
            raise ValueError(
                "active_references must be two canonical exact references"
            )
        if type(self.permutation_rank) is not int or self.permutation_rank not in {
            0,
            1,
        }:
            raise ValueError("two-slot permutation_rank must be exactly 0 or 1")
        if (
            type(self.randomization_policy_id) is not str
            or _TOKEN.fullmatch(self.randomization_policy_id) is None
        ):
            raise ValueError("randomization_policy_id must use the token grammar")
        if (
            type(self.randomization_policy_version) is not int
            or self.randomization_policy_version <= 0
        ):
            raise ValueError("randomization_policy_version must be positive")
        require_sha256(
            self.randomization_definition_sha256,
            "randomization_definition_sha256",
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_PERMUTATION_DOMAIN, self.to_record()),
        )

    @property
    def subset_ranks_by_slot(self) -> tuple[int, int]:
        return (0, 1) if self.permutation_rank == 0 else (1, 0)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "active_references": [
                {
                    "insight_id": reference.insight_id.value,
                    "version": reference.version,
                }
                for reference in self.active_references
            ],
            "permutation_rank": self.permutation_rank,
            "subset_ranks_by_slot": list(self.subset_ranks_by_slot),
            "randomization_policy_id": self.randomization_policy_id,
            "randomization_policy_version": self.randomization_policy_version,
            "randomization_definition_sha256": (
                self.randomization_definition_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class _ProspectiveEndpoint:
    slot_id: str
    reference: InsightRef | None
    option_id: str
    option_identity_sha256: str
    configuration: FrozenJsonValue
    configuration_sha256: str
    phenotype_identity_sha256: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProspectiveUnion:
    slot_id: str
    configuration: FrozenJsonValue
    configuration_sha256: str
    phenotype_identity_sha256: str
    prospective_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class G3ExpectedEndpoint:
    """Public immutable authority for one prospectively frozen G1/G2 endpoint."""

    slot_id: str
    reference: InsightRef | None
    option_id: str
    option_identity_sha256: str
    configuration: FrozenJsonValue
    configuration_sha256: str
    phenotype_identity_sha256: str
    changed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.slot_id) is not str or not self.slot_id:
            raise ValueError("endpoint slot_id must be non-empty exact text")
        if self.reference is not None:
            if type(self.reference) is not InsightRef:
                raise TypeError("endpoint reference must be an exact InsightRef")
            InsightRef.__post_init__(self.reference)
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("endpoint option_id must be non-empty exact text")
        require_sha256(
            self.option_identity_sha256,
            "option_identity_sha256",
        )
        if not is_frozen_json_value(self.configuration):
            raise TypeError("endpoint configuration must be frozen typed JSON")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        if typed_json_sha256(self.configuration) != self.configuration_sha256:
            raise ValueError("endpoint configuration hash does not authenticate value")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        if (
            type(self.changed_paths) is not tuple
            or any(type(value) is not str or not value for value in self.changed_paths)
            or self.changed_paths != tuple(sorted(set(self.changed_paths)))
        ):
            raise ValueError("endpoint changed_paths must be canonical and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "reference": (
                None
                if self.reference is None
                else {
                    "insight_id": self.reference.insight_id.value,
                    "version": self.reference.version,
                }
            ),
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "configuration_sha256": self.configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True, slots=True)
class G3ExpectedUnion:
    """Public prospective/runtime authority for one zero-call G3 union."""

    slot_id: str
    configuration: FrozenJsonValue
    configuration_sha256: str
    phenotype_identity_sha256: str
    prospective_materialization_receipt_sha256: str
    runtime_materialization_receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.slot_id) is not str or not self.slot_id:
            raise ValueError("union slot_id must be non-empty exact text")
        if not is_frozen_json_value(self.configuration):
            raise TypeError("union configuration must be frozen typed JSON")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        if typed_json_sha256(self.configuration) != self.configuration_sha256:
            raise ValueError("union configuration hash does not authenticate value")
        for name in (
            "phenotype_identity_sha256",
            "prospective_materialization_receipt_sha256",
            "runtime_materialization_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "configuration_sha256": self.configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "prospective_materialization_receipt_sha256": (
                self.prospective_materialization_receipt_sha256
            ),
            "runtime_materialization_receipt_sha256": (
                self.runtime_materialization_receipt_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class G3TerminalValidationAuthority:
    """Hash-bound expectations consumed by the post-G3 terminal gate.

    The planner creates this only after it has validated G1/G2 and constructed
    the exact zero-call G3 plan.  A feedback interceptor can therefore validate
    actual terminal outcomes without reaching into mutable planner internals.
    """

    hypothesis_parent_candidate_id: CandidateId
    hypothesis_parent_configuration: FrozenJsonValue
    hypothesis_parent_configuration_sha256: str
    hypothesis_parent_phenotype_identity_sha256: str
    seed_occurrence_binding_sha256: str
    seed_phenotype_identity_sha256s: tuple[str, str]
    g1_expected_endpoints: tuple[G3ExpectedEndpoint, G3ExpectedEndpoint]
    g2_expected_endpoints: tuple[
        G3ExpectedEndpoint,
        G3ExpectedEndpoint,
        G3ExpectedEndpoint,
        G3ExpectedEndpoint,
    ]
    g3_expected_unions: tuple[
        G3ExpectedUnion,
        G3ExpectedUnion,
        G3ExpectedUnion,
    ]
    prospective_proof_sha256: str
    g1_rendered_prompt_receipt_sha256: str
    g2_rendered_prompt_receipt_sha256: str
    genesis_snapshot_sha256: str
    diagnostic_wave_sha256: str
    closure_snapshot_sha256: str
    authority_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.hypothesis_parent_candidate_id) is not CandidateId:
            raise TypeError("hypothesis parent ID must be an exact CandidateId")
        CandidateId.__post_init__(self.hypothesis_parent_candidate_id)
        if not is_frozen_json_value(self.hypothesis_parent_configuration):
            raise TypeError("hypothesis parent configuration must be frozen JSON")
        require_sha256(
            self.hypothesis_parent_configuration_sha256,
            "hypothesis_parent_configuration_sha256",
        )
        if (
            typed_json_sha256(self.hypothesis_parent_configuration)
            != self.hypothesis_parent_configuration_sha256
        ):
            raise ValueError("hypothesis parent hash does not authenticate value")
        for name in (
            "hypothesis_parent_phenotype_identity_sha256",
            "seed_occurrence_binding_sha256",
            "prospective_proof_sha256",
            "g1_rendered_prompt_receipt_sha256",
            "g2_rendered_prompt_receipt_sha256",
            "genesis_snapshot_sha256",
            "diagnostic_wave_sha256",
            "closure_snapshot_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.seed_phenotype_identity_sha256s) is not tuple
            or len(self.seed_phenotype_identity_sha256s) != 2
        ):
            raise ValueError("terminal authority requires two seed phenotypes")
        for value in self.seed_phenotype_identity_sha256s:
            require_sha256(value, "seed phenotype identity")
        if len(set(self.seed_phenotype_identity_sha256s)) != 2:
            raise ValueError("seed phenotype identities must be distinct")
        endpoint_groups = (
            (self.g1_expected_endpoints, G1_DIAGNOSTIC_SLOT_IDS),
            (self.g2_expected_endpoints, G2_SLOT_IDS),
        )
        for endpoints, slot_ids in endpoint_groups:
            if type(endpoints) is not tuple or any(
                type(value) is not G3ExpectedEndpoint for value in endpoints
            ):
                raise TypeError("endpoint authorities must be exact values")
            if tuple(value.slot_id for value in endpoints) != slot_ids:
                raise ValueError("endpoint authority slot order changed")
        if type(self.g3_expected_unions) is not tuple or any(
            type(value) is not G3ExpectedUnion for value in self.g3_expected_unions
        ):
            raise TypeError("union authorities must be exact values")
        if tuple(value.slot_id for value in self.g3_expected_unions) != G3_SLOT_IDS[1:]:
            raise ValueError("union authority slot order changed")
        object.__setattr__(
            self,
            "authority_sha256",
            _hash(_PROSPECTIVE_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "hypothesis_parent_candidate_id": (
                self.hypothesis_parent_candidate_id.value
            ),
            "hypothesis_parent_configuration_sha256": (
                self.hypothesis_parent_configuration_sha256
            ),
            "hypothesis_parent_phenotype_identity_sha256": (
                self.hypothesis_parent_phenotype_identity_sha256
            ),
            "seed_occurrence_binding_sha256": self.seed_occurrence_binding_sha256,
            "seed_phenotype_identity_sha256s": list(
                self.seed_phenotype_identity_sha256s
            ),
            "g1_expected_endpoints": [
                value.to_record() for value in self.g1_expected_endpoints
            ],
            "g2_expected_endpoints": [
                value.to_record() for value in self.g2_expected_endpoints
            ],
            "g3_expected_unions": [
                value.to_record() for value in self.g3_expected_unions
            ],
            "prospective_proof_sha256": self.prospective_proof_sha256,
            "g1_rendered_prompt_receipt_sha256": (
                self.g1_rendered_prompt_receipt_sha256
            ),
            "g2_rendered_prompt_receipt_sha256": (
                self.g2_rendered_prompt_receipt_sha256
            ),
            "genesis_snapshot_sha256": self.genesis_snapshot_sha256,
            "diagnostic_wave_sha256": self.diagnostic_wave_sha256,
            "closure_snapshot_sha256": self.closure_snapshot_sha256,
        }


def _portable_compilation_record(
    request: HypothesisCompilationRequest,
    receipt: HypothesisCompilationReceipt,
) -> dict[str, object]:
    """Project a prepared compilation across placeholder occurrence IDs.

    Offline preparation cannot know the engine-assigned seed occurrence ID.
    Every other semantic input/output remains exact; the full prepared request
    and receipt hashes are retained separately in the matrix commitment.
    """

    validate_hypothesis_compilation(request, receipt)
    if not receipt.applicable or receipt.spec is None:
        raise ValueError("prepared G3 hypotheses must compile as applicable")
    spec = receipt.spec
    return {
        "reference": {
            "insight_id": request.reference.insight_id.value,
            "version": request.reference.version,
        },
        "insight_content_sha256": request.insight.content_sha256,
        "source_evidence_sha256": request.source_evidence_sha256,
        "requested_operator_kind": request.requested_operator_kind,
        "source_operator_kinds": list(request.source_operator_kinds),
        "parent_configuration_sha256": request.parent_configuration_sha256,
        "finite_contract_sha256": request.finite_contract.identity_sha256,
        "context_projection_sha256": request.context_projection_sha256,
        "endpoint_definition_sha256": request.endpoint_definition_sha256,
        "executable_operator_kinds": list(spec.executable_operator_kinds),
        "allowed_actions": [value.to_record() for value in spec.allowed_actions],
        "recommended_option_families": list(spec.recommended_option_families),
        "affected_paths": list(spec.affected_paths),
        "held_fixed_paths": list(spec.held_fixed_paths),
        "effect_predictions": [
            {
                "metric_id": value.metric_id,
                "direction": value.direction.value,
            }
            for value in spec.effect_predictions
        ],
        "falsification_condition": spec.falsification_condition,
        "compiler_policy_id": receipt.compiler_policy_id,
        "compiler_policy_version": receipt.compiler_policy_version,
        "compiler_definition_sha256": receipt.compiler_definition_sha256,
    }


@dataclass(frozen=True, slots=True)
class PreparedHypothesisMatrix:
    """Pre-run compiler authority for one parent role and exact card matrix."""

    parent_role: str
    requests: tuple[HypothesisCompilationRequest, HypothesisCompilationRequest]
    receipts: tuple[HypothesisCompilationReceipt, HypothesisCompilationReceipt]
    portable_matrix_sha256: str = field(init=False)
    commitment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.parent_role not in {"diagnostic_parent", "hypothesis_parent"}:
            raise ValueError("parent_role must be a frozen G3 parent role")
        if type(self.requests) is not tuple or len(self.requests) != 2:
            raise ValueError("prepared matrix requires two exact requests")
        if type(self.receipts) is not tuple or len(self.receipts) != 2:
            raise ValueError("prepared matrix requires two exact receipts")
        if any(type(value) is not HypothesisCompilationRequest for value in self.requests):
            raise TypeError("prepared requests must be exact")
        if any(type(value) is not HypothesisCompilationReceipt for value in self.receipts):
            raise TypeError("prepared receipts must be exact")
        portable = tuple(
            _portable_compilation_record(request, receipt)
            for request, receipt in zip(self.requests, self.receipts, strict=True)
        )
        references = tuple(request.reference for request in self.requests)
        if references != tuple(sorted(set(references))):
            raise ValueError("prepared matrix references must be canonical and unique")
        shared = {
            (
                request.parent_configuration_sha256,
                request.finite_contract.identity_sha256,
                request.context_projection_sha256,
                request.endpoint_definition_sha256,
                request.requested_operator_kind,
            )
            for request in self.requests
        }
        if len(shared) != 1:
            raise ValueError("prepared matrix mixes parent execution contexts")
        portable_sha256 = _hash(_PROSPECTIVE_DOMAIN, list(portable))
        object.__setattr__(self, "portable_matrix_sha256", portable_sha256)
        object.__setattr__(
            self,
            "commitment_sha256",
            _hash(
                _PROSPECTIVE_DOMAIN,
                {
                    "schema_version": 1,
                    "parent_role": self.parent_role,
                    "portable_matrix_sha256": portable_sha256,
                    "prepared_request_sha256s": [
                        request.request_sha256 for request in self.requests
                    ],
                    "prepared_receipt_sha256s": [
                        receipt.receipt_sha256 for receipt in self.receipts
                    ],
                },
            ),
        )

    @property
    def references(self) -> tuple[InsightRef, InsightRef]:
        return self.requests[0].reference, self.requests[1].reference

    @property
    def parent_configuration_sha256(self) -> str:
        return self.requests[0].parent_configuration_sha256

    def validate_runtime(
        self,
        matrix: tuple[CompiledHypothesisTreatment, ...],
    ) -> None:
        self.__post_init__()
        if type(matrix) is not tuple or len(matrix) != 2:
            raise ValueError("runtime hypothesis matrix must contain two treatments")
        portable = tuple(
            _portable_compilation_record(value.request, value.receipt)
            for value in matrix
        )
        observed_sha256 = _hash(_PROSPECTIVE_DOMAIN, list(portable))
        if observed_sha256 != self.portable_matrix_sha256:
            raise ValueError(
                "runtime hypothesis compilation differs from pre-run authority"
            )


@runtime_checkable
class G3BenchmarkBoundary(Protocol):
    """Narrow inverted boundary implemented by the public benchmark bundle."""

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract: ...

    def compile_registered_hypothesis_treatment(
        self,
        *,
        catalog_id: str,
        parent_candidate_id: CandidateId,
        parent_configuration: object,
        entry: InsightMemoryEntry,
        requested_operator_kind: str,
        context_projection_sha256: str,
        endpoint_definition_sha256: str,
    ) -> CompiledHypothesisTreatment: ...


def finite_mutation_boundary(
    *,
    contract: FiniteVariationContract,
    parent_candidate_id: CandidateId,
) -> tuple[tuple[str, ...], MutationContract]:
    """Derive the smallest complete machine boundary for a finite palette."""

    validate_finite_variation_contract(contract)
    probe = CandidateId("candidate_g3_finite_boundary_probe")
    if probe == parent_candidate_id:
        probe = CandidateId("candidate_g3_finite_boundary_probe_alternate")
    paths: dict[bytes, JsonPath] = {}
    max_changed_paths = 0
    max_operations = 0
    for option in contract.options:
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=parent_candidate_id,
            target_candidate_id=probe,
        )
        changed = {operation.path for operation in patch.operations}
        max_changed_paths = max(max_changed_paths, len(changed))
        max_operations = max(max_operations, len(patch.operations))
        for path in changed:
            paths[canonical_path_bytes(path)] = path
    editable = tuple(paths[key] for key in sorted(paths))
    if not editable or max_changed_paths <= 0 or max_operations <= 0:
        raise ValueError("finite variation palette has no executable mutation")
    allowed = tuple(
        sorted(
            {
                path.segments[0].value
                for path in editable
                if type(path.segments[0]) is ObjectKey
            }
        )
    )
    return allowed, MutationContract(
        editable_paths=editable,
        max_changed_paths=max_changed_paths,
        max_operations=max_operations,
        allow_abstention=False,
    )


def _materialized_finite_choice(
    *,
    ids: IdFactory,
    parent: EvolutionCandidate,
    generation: int,
    label: str,
    contract: FiniteVariationContract,
    choice: ParentBoundActionChoice,
) -> MaterializedInvocation:
    choice.validate_contract(contract)
    option = contract.resolve(choice.option_id)
    probe = ids.new_candidate_id()
    patch = derive_patch(
        parent.configuration,
        option.child_configuration,
        base_candidate_id=parent.candidate_id,
        target_candidate_id=probe,
    )
    paths = tuple(sorted({_path_text(operation.path) for operation in patch.operations}))
    top_level = tuple(
        sorted(
            {
                operation.path.segments[0].value
                for operation in patch.operations
                if type(operation.path.segments[0]) is ObjectKey
            }
        )
    )
    plan = InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=generation,
        label=label,
        allowed_top_level=top_level,
        phase="g3_engine_mate",
    )
    configuration = thaw_json(option.child_configuration)
    if type(configuration) is not dict:
        raise TypeError("finite choice child must be an object")
    return MaterializedInvocation(
        plan=plan,
        draft=CandidateDraft(
            configuration=configuration,
            design_rationale="Engine-owned outcome-blind parent-bound mate action.",
            intended_changes=paths,
            source_attribution=tuple(
                SourceAttribution(path, "mutation") for path in paths
            ),
        ),
        candidate_id=probe,
        materialization_policy_id=choice.selection_policy_id,
        materialization_policy_version=choice.selection_policy_version,
        materialization_receipt_hash=choice.choice_sha256,
    )


def _neutral_sham_requirement(
    *,
    entry: InsightMemoryEntry,
    contract: FiniteVariationContract,
    choice: ParentBoundActionChoice,
) -> InsightTreatmentRequirement:
    choice.validate_contract(contract)
    if entry.lifecycle_state is not InsightLifecycleState.QUARANTINED:
        raise ValueError("neutral sham card must remain quarantined")
    if entry.evidence_lineage is not None or entry.draft.evidence_contrast_ids:
        raise ValueError("neutral sham card must be evidence-free")
    if any(
        prediction.direction is not MetricEffectDirection.UNKNOWN
        for prediction in entry.draft.effect_predictions
    ):
        raise ValueError("neutral sham card cannot make a directional prediction")
    option = contract.resolve(choice.option_id)
    if entry.draft.recommended_option_ids != (option.option_id,):
        raise ValueError("neutral sham card must name its exact parent-bound option")
    if entry.draft.recommended_option_families != (option.family,):
        raise ValueError("neutral sham card family differs from its exact option")
    evidence = TreatmentInsightEvidence(
        reference=entry.reference,
        insight_content_sha256=entry.draft.content_sha256,
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        affected_paths=tuple(sorted(entry.draft.affected_paths)),
        recommended_option_families=entry.draft.recommended_option_families,
        recommended_option_ids=entry.draft.recommended_option_ids,
    )
    return InsightTreatmentRequirement(
        insight_bindings=(evidence.binding(),),
        finite_contract_sha256=contract.identity_sha256,
        allowed_actions=(
            TreatmentActionBinding(option.option_id, option.identity_sha256),
        ),
        claim_mode=TreatmentClaimMode.EXACT_REQUIRED,
        assignment_role=TreatmentAssignmentRole.SHAM_CONTROL,
        require_option_family_match=True,
        require_changed_path_overlap=True,
    )


class G3CausalScreenPlanner:
    """Stateful deterministic planner for the exact three-wave screen."""

    policy_id = G3_SCREEN_POLICY_ID
    policy_version = G3_SCREEN_POLICY_VERSION

    def __init__(
        self,
        *,
        benchmark: G3BenchmarkBoundary,
        engine: AgenticEvolutionEngine,
        ids: IdFactory,
        memory: InsightMemoryBank,
        reward_binding: RewardPolicyBinding,
        active_references: tuple[InsightRef, InsightRef],
        neutral_reference: InsightRef,
        diagnostic_permutation: FrozenDiagnosticPermutation,
        prepared_hypothesis_matrices: tuple[
            PreparedHypothesisMatrix,
            PreparedHypothesisMatrix,
        ],
        model_catalog_id: str,
        neutral_choice: ParentBoundActionChoice,
        mate_choice: ParentBoundActionChoice,
        diagnostic_parent_configuration_sha256: str,
        hypothesis_parent_configuration_sha256: str,
        endpoint_definition_sha256: str,
        estimand_stratum_sha256: str,
        phase: str = "g3_causal_screen",
        no_yield_reward: float = -1.0,
        score_policy: CausalSearchScorePolicy | None = None,
        controls: DeterministicMemoryControlPolicy | None = None,
        trace_sink=None,
    ) -> None:
        if not isinstance(benchmark, G3BenchmarkBoundary):
            raise TypeError("benchmark must implement G3BenchmarkBoundary")
        if not isinstance(engine, AgenticEvolutionEngine):
            raise TypeError("engine must be an AgenticEvolutionEngine")
        if not isinstance(ids, IdFactory):
            raise TypeError("ids must implement IdFactory")
        if type(memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if type(reward_binding) is not RewardPolicyBinding:
            raise TypeError("reward_binding must be exact")
        RewardPolicyBinding.__post_init__(reward_binding)
        if endpoint_definition_sha256 != reward_binding.definition_hash:
            raise ValueError(
                "endpoint_definition_sha256 must equal the active reward/Q definition"
            )
        if (
            type(active_references) is not tuple
            or len(active_references) != 2
            or active_references != tuple(sorted(set(active_references)))
        ):
            raise ValueError("active_references must be two canonical exact refs")
        if neutral_reference in active_references:
            raise ValueError("neutral sham must be distinct from active hypotheses")
        if type(diagnostic_permutation) is not FrozenDiagnosticPermutation:
            raise TypeError("diagnostic_permutation must be exact")
        FrozenDiagnosticPermutation.__post_init__(diagnostic_permutation)
        if diagnostic_permutation.active_references != active_references:
            raise ValueError("diagnostic permutation differs from active references")
        if (
            type(prepared_hypothesis_matrices) is not tuple
            or len(prepared_hypothesis_matrices) != 2
            or any(
                type(value) is not PreparedHypothesisMatrix
                for value in prepared_hypothesis_matrices
            )
        ):
            raise TypeError("prepared_hypothesis_matrices must contain two matrices")
        for value in prepared_hypothesis_matrices:
            PreparedHypothesisMatrix.__post_init__(value)
        if tuple(value.parent_role for value in prepared_hypothesis_matrices) != (
            "diagnostic_parent",
            "hypothesis_parent",
        ):
            raise ValueError("prepared matrices must use frozen G3 parent order")
        if any(
            value.references != active_references
            for value in prepared_hypothesis_matrices
        ):
            raise ValueError("prepared matrices differ from active references")
        for value in (
            diagnostic_parent_configuration_sha256,
            hypothesis_parent_configuration_sha256,
            endpoint_definition_sha256,
            estimand_stratum_sha256,
        ):
            require_sha256(value, "g3 screen identity")
        prepared_parent_hashes = tuple(
            value.parent_configuration_sha256
            for value in prepared_hypothesis_matrices
        )
        if prepared_parent_hashes != (
            diagnostic_parent_configuration_sha256,
            hypothesis_parent_configuration_sha256,
        ):
            raise ValueError("prepared matrices differ from frozen G3 parents")
        if any(
            request.endpoint_definition_sha256 != endpoint_definition_sha256
            for matrix in prepared_hypothesis_matrices
            for request in matrix.requests
        ):
            raise ValueError("prepared matrices differ from the frozen G3 endpoint")
        if type(model_catalog_id) is not str or _TOKEN.fullmatch(model_catalog_id) is None:
            raise ValueError("model_catalog_id must use the token grammar")
        if neutral_choice.catalog_id != model_catalog_id:
            raise ValueError("neutral action choice must use the model catalog")
        if neutral_choice.role != "neutral_sham" or mate_choice.role != "orthogonal_mate":
            raise ValueError("action choices have incorrect G3 roles")
        if type(phase) is not str or not phase.strip():
            raise ValueError("phase must be non-empty")
        if type(no_yield_reward) is not float or not math.isfinite(no_yield_reward):
            raise TypeError("no_yield_reward must be a finite canonical float")
        if no_yield_reward != reward_binding.failure_score:
            raise ValueError(
                "G3 no_yield_reward must equal the active reward failure score"
            )

        self.benchmark = benchmark
        self.engine = engine
        self.ids = ids
        self.memory = memory
        self.reward_binding = reward_binding
        self.active_references = active_references
        self.neutral_reference = neutral_reference
        self.diagnostic_permutation = diagnostic_permutation
        self.prepared_hypothesis_matrices = prepared_hypothesis_matrices
        self.model_catalog_id = model_catalog_id
        self.neutral_choice = neutral_choice
        self.mate_choice = mate_choice
        self.diagnostic_parent_configuration_sha256 = (
            diagnostic_parent_configuration_sha256
        )
        self.hypothesis_parent_configuration_sha256 = (
            hypothesis_parent_configuration_sha256
        )
        self.endpoint_definition_sha256 = endpoint_definition_sha256
        self.estimand_stratum_sha256 = estimand_stratum_sha256
        self.phase = phase
        self.no_yield_reward = no_yield_reward
        self.score_policy = score_policy or CausalSearchScorePolicy(
            prior_effective_sample_size=1.0,
            uncertainty_scale=0.0,
            exploration_weight=0.0,
        )
        self.controls = controls or DeterministicMemoryControlPolicy()
        self.checkpoint_service = DiagnosticMemoryCheckpointService(
            WaveSealedCheckpointBuilder(self.score_policy),
            trace_sink=trace_sink,
        )
        self.trace_sink = trace_sink
        self.genesis = None
        self.wave: FrozenDiagnosticMemoryWave | None = None
        self.closure: MemoryCheckpointClosure | None = None
        self.g1_prompt_shape_sha256: str | None = None
        self.g2_prompt_shape_sha256: str | None = None
        self.g1_rendered_prompt_receipt: MatchedPromptStructureReceipt | None = None
        self.g2_rendered_prompt_receipt: MatchedPromptStructureReceipt | None = None
        self.g2_assignments: tuple[ResolvedInsightAssignment, ...] = ()
        self._diagnostic_parent_id: CandidateId | None = None
        self._hypothesis_parent_id: CandidateId | None = None
        self._seed_occurrence_binding_sha256: str | None = None
        self._seed_phenotype_sha256s: tuple[str, str] | None = None
        self._runtime_diagnostic_matrix: tuple[
            CompiledHypothesisTreatment,
            ...,
        ] = ()
        self._runtime_hypothesis_matrix: tuple[
            CompiledHypothesisTreatment,
            ...,
        ] = ()
        self._g1_expected: tuple[_ProspectiveEndpoint, ...] = ()
        self._g2_expected: tuple[_ProspectiveEndpoint, ...] = ()
        self._g2_prospective_unions: tuple[_ProspectiveUnion, ...] = ()
        self._g2_prospective_proof_sha256: str | None = None
        self._terminal_validation_authority: (
            G3TerminalValidationAuthority | None
        ) = None

    @property
    def terminal_validation_authority(
        self,
    ) -> G3TerminalValidationAuthority | None:
        """Return the immutable post-G3 authority once the G3 plan is frozen."""

        return self._terminal_validation_authority

    def _reward(self, state: OptimizerState, generation: int) -> FrozenWaveReward:
        return FrozenWaveReward(
            binding=self.reward_binding,
            archive_snapshot_hash=state.archive_snapshot_hash,
            reward_snapshot_hash=_hash(
                b"agent-evolve:g3-wave-reward:v1\x00",
                {
                    "generation": generation,
                    "archive_snapshot_hash": state.archive_snapshot_hash,
                    "endpoint_definition_sha256": self.endpoint_definition_sha256,
                },
            ),
        )

    def plan(self, state: OptimizerState, budget: OptimizerBudget) -> GenerationPlan:
        if budget != G3_SCREEN_BUDGET:
            raise ValueError("G3 screen requires the exact 6-call/11-evaluation budget")
        generation = state.generation + 1
        if generation == 1:
            return self._g1(state)
        if generation == 2:
            return self._g2(state)
        if generation == 3:
            return self._g3(state)
        raise ValueError("G3 causal screen has exactly three generations")

    @staticmethod
    def _occurrence_record(candidate: EvolutionCandidate) -> dict[str, object]:
        occurrence = candidate.occurrence
        return {
            "candidate_id": occurrence.candidate_id.value,
            "configuration_hash": occurrence.configuration_hash,
            "configuration_artifact_hash": (
                occurrence.configuration_artifact_hash
            ),
            "proposal_sequence": occurrence.proposal_sequence,
            "operator_invocation_id": (
                None
                if occurrence.operator_invocation_id is None
                else occurrence.operator_invocation_id.value
            ),
        }

    def _phenotype_sha256(self, candidate: EvolutionCandidate) -> str:
        identity = self.engine.identify_phenotype(candidate)
        detailed = candidate.detailed_evaluation
        if detailed is not None:
            if not detailed.success:
                raise ValueError("G3 endpoint detailed evaluation did not succeed")
            if detailed.phenotype != identity:
                raise ValueError(
                    "candidate detailed phenotype differs from engine policy"
                )
        return identity.identity_sha256

    def _parents(
        self,
        state: OptimizerState,
    ) -> tuple[EvolutionCandidate, EvolutionCandidate]:
        diagnostic, hypothesis = state.candidates[:2]
        if diagnostic.occurrence.configuration_hash != (
            self.diagnostic_parent_configuration_sha256
        ):
            raise ValueError("diagnostic seed differs from frozen G3 parent")
        if hypothesis.occurrence.configuration_hash != (
            self.hypothesis_parent_configuration_sha256
        ):
            raise ValueError("hypothesis seed differs from frozen G3 parent")
        if any(
            not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
            for candidate in (diagnostic, hypothesis)
        ):
            raise ValueError("G3 seeds must be valid and per-protocol")
        observed_ids = (diagnostic.candidate_id, hypothesis.candidate_id)
        occurrence_sha256 = _hash(
            _OCCURRENCE_DOMAIN,
            [
                self._occurrence_record(diagnostic),
                self._occurrence_record(hypothesis),
            ],
        )
        phenotype_sha256s = (
            self._phenotype_sha256(diagnostic),
            self._phenotype_sha256(hypothesis),
        )
        if len(set(phenotype_sha256s)) != 2:
            raise ValueError("G3 seeds collide under semantic phenotype identity")
        if self._diagnostic_parent_id is None:
            if state.generation != 0:
                raise RuntimeError("seed occurrences were not frozen before G1")
            self._diagnostic_parent_id, self._hypothesis_parent_id = observed_ids
            self._seed_occurrence_binding_sha256 = occurrence_sha256
            self._seed_phenotype_sha256s = phenotype_sha256s
        elif (
            observed_ids
            != (self._diagnostic_parent_id, self._hypothesis_parent_id)
            or occurrence_sha256 != self._seed_occurrence_binding_sha256
            or phenotype_sha256s != self._seed_phenotype_sha256s
        ):
            raise ValueError("frozen G3 seed occurrences changed")
        return diagnostic, hypothesis

    def _require_exact_state(self, state: OptimizerState) -> None:
        expected = {
            0: (2, 0, 0, 2, 0),
            1: (4, 1, 1, 4, 2),
            2: (8, 2, 2, 8, 5),
        }.get(state.generation)
        if expected is None:
            raise ValueError("G3 planner received an unsupported generation state")
        (
            candidate_count,
            generation_receipt_count,
            feedback_receipt_count,
            unique_evaluations,
            logical_llm_calls,
        ) = expected
        observed = (
            len(state.candidates),
            len(state.generation_receipts),
            len(state.feedback_receipts),
            state.unique_evaluations,
            state.logical_llm_calls,
        )
        if observed != expected:
            raise ValueError(
                "G3 state differs from the exact 2-to-4-to-8 causal protocol"
            )
        for receipt in state.feedback_receipts:
            if receipt.used_logical_llm_calls != 0:
                raise ValueError("G1/G2 feedback must be a zero-call sealed no-op")
        if state.generation >= 1:
            first = state.generation_receipts[0]
            if (
                first.logical_llm_calls_before,
                first.logical_llm_calls_after,
                first.unique_evaluations_before,
                first.unique_evaluations_after,
            ) != (0, 2, 2, 4):
                raise ValueError("G1 counters differ from two calls/two fresh misses")
        if state.generation >= 2:
            second = state.generation_receipts[1]
            if (
                second.logical_llm_calls_before,
                second.logical_llm_calls_after,
                second.unique_evaluations_before,
                second.unique_evaluations_after,
            ) != (2, 5, 4, 8):
                raise ValueError("G2 counters differ from three calls/four fresh misses")
        self._parents(state)

    @staticmethod
    def _probe_candidate_id(label: str, forbidden: set[CandidateId]) -> CandidateId:
        # Slot labels are durable scientific metadata and may intentionally use
        # words that the identifier policy forbids as embedded content markers.
        # Keep the prospective lineage identity opaque while deterministically
        # binding it to the exact slot label.
        opaque_label = hashlib.sha256(
            label.encode("utf-8", errors="strict")
        ).hexdigest()[:16]
        base = f"candidate_g3_probe_{opaque_label}"
        for suffix in ("", "_alternate", "_second_alternate"):
            value = CandidateId(base + suffix)
            if value not in forbidden:
                return value
        raise RuntimeError("cannot allocate a prospective candidate identity")

    def _endpoint(
        self,
        *,
        slot_id: str,
        reference: InsightRef | None,
        parent: EvolutionCandidate,
        contract: FiniteVariationContract,
        option_id: str,
    ) -> _ProspectiveEndpoint:
        option = contract.resolve(option_id)
        target = self._probe_candidate_id(slot_id, {parent.candidate_id})
        patch = derive_patch(
            parent.configuration,
            option.child_configuration,
            base_candidate_id=parent.candidate_id,
            target_candidate_id=target,
        )
        if not patch.operations:
            raise ValueError("G3 treatment option is an empty parent-relative action")
        paths = tuple(
            sorted({_path_text(operation.path) for operation in patch.operations})
        )
        phenotype = self.engine.identify_phenotype(option.child_configuration)
        return _ProspectiveEndpoint(
            slot_id=slot_id,
            reference=reference,
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            configuration=option.child_configuration,
            configuration_sha256=option.child_configuration_sha256,
            phenotype_identity_sha256=phenotype.identity_sha256,
            changed_paths=paths,
        )

    def _base_model_plan(
        self,
        *,
        parent: EvolutionCandidate,
        generation: int,
        label: str,
        contract: FiniteVariationContract,
    ) -> InvocationPlan:
        allowed, mutation = finite_mutation_boundary(
            contract=contract,
            parent_candidate_id=parent.candidate_id,
        )
        return InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=generation,
            label=label,
            allowed_top_level=allowed,
            mutation_contract=mutation,
            mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
            finite_variation_contract=contract,
            phase=self.phase,
        )

    def _compile_matrix(
        self,
        *,
        parent: EvolutionCandidate,
        context_sha256: str,
        prepared: PreparedHypothesisMatrix,
    ) -> tuple[CompiledHypothesisTreatment, ...]:
        entries = self.memory.entries_for(self.active_references)
        if any(
            entry.lifecycle_state is InsightLifecycleState.DEPRECATED
            for entry in entries
        ):
            raise ValueError("deprecated hypotheses cannot enter a G3 treatment")
        compiled = tuple(
            self.benchmark.compile_registered_hypothesis_treatment(
                catalog_id=self.model_catalog_id,
                parent_candidate_id=parent.candidate_id,
                parent_configuration=parent.configuration,
                entry=entry,
                requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
                context_projection_sha256=context_sha256,
                endpoint_definition_sha256=self.endpoint_definition_sha256,
            )
            for entry in entries
        )
        if tuple(value.request.reference for value in compiled) != self.active_references:
            raise RuntimeError("compiled hypothesis matrix changed reference order")
        if any(len(value.requirement.allowed_actions) != 1 for value in compiled):
            raise ValueError("G3 hypotheses must compile to exact singleton actions")
        actions = tuple(
            value.requirement.allowed_actions[0].option_identity_sha256
            for value in compiled
        )
        if len(set(actions)) != len(actions):
            raise ValueError("active hypotheses compiled to the same exact action")
        prepared.validate_runtime(compiled)
        return compiled

    def _g1(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is not None:
            raise RuntimeError("G1 diagnostic wave was already frozen")
        self._require_exact_state(state)
        diagnostic, hypothesis = self._parents(state)
        contract = self.benchmark.bind_finite_variation(
            self.model_catalog_id,
            diagnostic.configuration,
        )
        hypothesis_contract = self.benchmark.bind_finite_variation(
            self.model_catalog_id,
            hypothesis.configuration,
        )
        base = self._base_model_plan(
            parent=diagnostic,
            generation=1,
            label="g1_diagnostic",
            contract=contract,
        )
        context_sha256 = context_stratum_hash(
            problem_id=self.engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=self.phase,
        )
        self._runtime_diagnostic_matrix = self._compile_matrix(
            parent=diagnostic,
            context_sha256=context_sha256,
            prepared=self.prepared_hypothesis_matrices[0],
        )
        self._runtime_hypothesis_matrix = self._compile_matrix(
            parent=hypothesis,
            context_sha256=context_sha256,
            prepared=self.prepared_hypothesis_matrices[1],
        )
        if any(
            value.request.finite_contract.identity_sha256 != contract.identity_sha256
            for value in self._runtime_diagnostic_matrix
        ) or any(
            value.request.finite_contract.identity_sha256
            != hypothesis_contract.identity_sha256
            for value in self._runtime_hypothesis_matrix
        ):
            raise ValueError("compiled runtime matrices differ from bound catalogs")
        self._g1_expected = tuple(
            self._endpoint(
                slot_id=G1_DIAGNOSTIC_SLOT_IDS[index],
                reference=value.request.reference,
                parent=diagnostic,
                contract=contract,
                option_id=value.requirement.allowed_actions[0].option_id,
            )
            for index, value in enumerate(self._runtime_diagnostic_matrix)
        )
        g1_phenotypes = tuple(
            value.phenotype_identity_sha256 for value in self._g1_expected
        )
        if len(set(g1_phenotypes)) != 2 or set(g1_phenotypes).intersection(
            self._seed_phenotype_sha256s or ()
        ):
            raise ValueError("G1 hypotheses do not define two fresh phenotypes")
        entries = self.memory.entries_for(self.active_references)
        self.genesis = self.score_policy.genesis(
            exact_context_hash=context_sha256,
            estimand_stratum_hash=self.estimand_stratum_sha256,
            priors={entry.reference: entry.initial_score for entry in entries},
        )
        self.g1_prompt_shape_sha256 = self.engine.prompt_shape_commitment(
            base,
            selected_insight_count=1,
            reward_definition_hash=self.reward_binding.definition_hash,
        )
        assignments = tuple(
            ResolvedInsightAssignment.resolve(
                credit_unit_id=self.ids.new_operator_invocation_id(),
                snapshot=self.genesis,
                expected_snapshot_sha256=self.genesis.snapshot_sha256,
                block_id="g1_diagnostic_randomized_block",
                arm=MemoryAssignmentArm.DIAGNOSTIC,
                selection_decision=self.controls.uniform(
                    snapshot=self.genesis,
                    subset_size=1,
                    subset_rank=rank,
                ),
                prompt_shape_sha256=self.g1_prompt_shape_sha256,
            )
            for rank in self.diagnostic_permutation.subset_ranks_by_slot
        )
        if tuple(
            assignment.selection_decision.selected[0]
            for assignment in assignments
        ) != tuple(
            self.active_references[rank]
            for rank in self.diagnostic_permutation.subset_ranks_by_slot
        ):
            raise RuntimeError("joint diagnostic permutation realization drifted")
        self.wave = FrozenDiagnosticMemoryWave(
            wave_id="g3_causal_screen_diagnostic_wave",
            prior_snapshot=self.genesis,
            assignments=tuple(sorted(assignments, key=lambda value: value.assignment_sha256)),
            reward_definition_hash=self.reward_binding.definition_hash,
            no_yield_reward=self.no_yield_reward,
        )
        self.checkpoint_service.publish_frozen_wave(self.wave)
        matrix = self._runtime_diagnostic_matrix
        by_ref = {value.request.reference: value for value in matrix}
        expected_by_ref = {value.reference: value for value in self._g1_expected}
        slots = tuple(
            OptimizerSlot.model(
                slot_id=G1_DIAGNOSTIC_SLOT_IDS[index],
                role="diagnostic_active_hypothesis",
                plan=replace(
                    base,
                    label=G1_DIAGNOSTIC_SLOT_IDS[index],
                    resolved_insight_assignment=assignment,
                    insight_treatment_requirement=(
                        by_ref[assignment.selection_decision.selected[0]].requirement
                    ),
                    compiled_hypothesis_treatment=(
                        by_ref[assignment.selection_decision.selected[0]]
                    ),
                    compiled_hypothesis_eligibility=matrix,
                ),
            )
            for index, assignment in enumerate(assignments)
        )
        for slot, assignment in zip(slots, assignments, strict=True):
            selected = assignment.selection_decision.selected[0]
            expected = expected_by_ref[selected]
            if (
                slot.plan.insight_treatment_requirement.allowed_actions[0].option_id
                != expected.option_id
            ):
                raise RuntimeError("G1 slot differs from its prospective endpoint")
        return GenerationPlan(
            generation=1,
            slots=slots,
            reward=self._reward(state, 1),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("diagnostic_permutation_receipt_sha256", self.diagnostic_permutation.receipt_sha256),
                        ("diagnostic_wave_sha256", self.wave.wave_sha256),
                        ("hypothesis_runtime_matrix_sha256", _hash(_PROSPECTIVE_DOMAIN, [value.binding_sha256 for value in self._runtime_hypothesis_matrix])),
                        ("prepared_diagnostic_matrix_sha256", self.prepared_hypothesis_matrices[0].commitment_sha256),
                        ("prepared_hypothesis_matrix_sha256", self.prepared_hypothesis_matrices[1].commitment_sha256),
                        ("prompt_shape_sha256", self.g1_prompt_shape_sha256),
                        ("seed_occurrence_binding_sha256", self._seed_occurrence_binding_sha256),
                    )
                )
            ),
        )

    def _require_model_endpoint(
        self,
        outcome: InvocationOutcome,
        *,
        expected: _ProspectiveEndpoint,
        assignment_role: TreatmentAssignmentRole,
        assignment_kind: InsightAssignmentKind,
        generation: int,
    ) -> EvolutionCandidate:
        if outcome.failure_stage is not None or outcome.candidate is None:
            raise ValueError("G3 model treatment did not complete successfully")
        prepared = outcome.prepared
        plan = prepared.plan
        candidate = outcome.candidate
        if (
            prepared.proposal_authority is not ProposalAuthority.MODEL
            or prepared.call_id is None
            or plan.operator_kind is not OperatorKind.TYPED_MUTATION
            or candidate.operator_kind is not OperatorKind.TYPED_MUTATION
        ):
            raise ValueError("G3 model endpoint has the wrong proposal authority")
        if candidate.generation != generation:
            raise ValueError("G3 model endpoint has the wrong generation")
        if (
            not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
        ):
            raise ValueError("G3 model endpoint is invalid or noncompliant")
        if candidate.occurrence.operator_invocation_id != prepared.operator_invocation_id:
            raise ValueError("G3 endpoint occurrence differs from its invocation")
        requirement = plan.insight_treatment_requirement
        if requirement is None or requirement.assignment_role is not assignment_role:
            raise ValueError("G3 endpoint has the wrong treatment role")
        if len(requirement.allowed_actions) != 1:
            raise ValueError("G3 endpoint treatment is not an exact singleton")
        action_binding = requirement.allowed_actions[0]
        if (
            action_binding.option_id,
            action_binding.option_identity_sha256,
        ) != (expected.option_id, expected.option_identity_sha256):
            raise ValueError("G3 endpoint differs from its frozen exact action")
        preflight = prepared.treatment_preflight_receipt
        if (
            preflight is None
            or not preflight.passed
            or len(preflight.compatible_actions) != 1
            or preflight.compatible_actions[0].binding() != action_binding
        ):
            raise ValueError("G3 treatment preflight did not admit one exact action")
        admission = outcome.treatment_admission_receipt
        if (
            admission is None
            or not admission.passed
            or admission.selected_action.binding() != action_binding
        ):
            raise ValueError("G3 treatment admission did not pass exactly")
        reference = expected.reference
        if reference is None:
            raise RuntimeError("model endpoint lost its treatment reference")
        if (
            candidate.selected_insight_refs != (reference,)
            or candidate.claimed_insight_ids != (reference.insight_id.value,)
            or candidate.insight_assignment_kind is not assignment_kind
        ):
            raise ValueError("G3 endpoint did not instantiate its assigned card")
        if candidate.occurrence.configuration_hash != expected.configuration_sha256:
            raise ValueError("G3 endpoint configuration differs from frozen action")
        if not typed_json_equal(candidate.configuration, expected.configuration):
            raise ValueError("G3 endpoint typed configuration changed")
        if self._phenotype_sha256(candidate) != expected.phenotype_identity_sha256:
            raise ValueError("G3 endpoint semantic phenotype changed")
        if assignment_role is TreatmentAssignmentRole.ACTIVE:
            if (
                plan.resolved_insight_assignment is None
                or plan.compiled_hypothesis_treatment is None
                or not plan.compiled_hypothesis_eligibility
            ):
                raise ValueError("active G3 treatment lost compiled causal authority")
        elif (
            plan.resolved_insight_assignment is not None
            or plan.compiled_hypothesis_treatment is not None
            or plan.compiled_hypothesis_eligibility
            or plan.quarantine_test_insights != (reference,)
        ):
            raise ValueError("sham G3 endpoint acquired causal-memory authority")
        return candidate

    def _require_engine_endpoint(
        self,
        outcome: InvocationOutcome,
        *,
        expected: _ProspectiveEndpoint,
        generation: int,
    ) -> EvolutionCandidate:
        if outcome.failure_stage is not None or outcome.candidate is None:
            raise ValueError("G3 engine endpoint did not complete successfully")
        prepared = outcome.prepared
        candidate = outcome.candidate
        if (
            prepared.proposal_authority is not ProposalAuthority.ENGINE
            or prepared.call_id is not None
            or prepared.plan.operator_kind is not OperatorKind.TYPED_MUTATION
            or candidate.operator_kind is not OperatorKind.TYPED_MUTATION
            or outcome.treatment_admission_receipt is not None
        ):
            raise ValueError("G3 mate has the wrong engine-only authority")
        if (
            candidate.generation != generation
            or not candidate.valid
            or not candidate.operator_compliant
            or not candidate.evidence_compliant
        ):
            raise ValueError("G3 mate is invalid or noncompliant")
        if candidate.occurrence.operator_invocation_id != prepared.operator_invocation_id:
            raise ValueError("G3 mate occurrence differs from its invocation")
        if (
            candidate.occurrence.configuration_hash != expected.configuration_sha256
            or not typed_json_equal(candidate.configuration, expected.configuration)
            or self._phenotype_sha256(candidate)
            != expected.phenotype_identity_sha256
        ):
            raise ValueError("G3 mate differs from its frozen prospective endpoint")
        return candidate

    @staticmethod
    def _prompt_receipt(
        receipt: GenerationReceipt,
        slot_ids: tuple[str, ...],
    ) -> MatchedPromptStructureReceipt:
        by_slot = {value.slot.slot_id: value.outcome for value in receipt.slot_results}
        if tuple(by_slot) != tuple(value.slot.slot_id for value in receipt.slot_results):
            raise ValueError("generation receipt repeats or reorders slot IDs")
        return seal_matched_prompt_structure(
            tuple(by_slot[slot_id].prepared.prompt for slot_id in slot_ids)
        )

    def _prospective_union(
        self,
        *,
        hypothesis: EvolutionCandidate,
        model_endpoint: _ProspectiveEndpoint,
        mate_endpoint: _ProspectiveEndpoint,
        slot_id: str,
    ) -> tuple[_ProspectiveUnion, DisjointPatchMaterialization]:
        forbidden = {hypothesis.candidate_id}
        left_id = self._probe_candidate_id(f"{slot_id}_left", forbidden)
        forbidden.add(left_id)
        right_id = self._probe_candidate_id(f"{slot_id}_right", forbidden)
        forbidden.add(right_id)
        target_id = self._probe_candidate_id(f"{slot_id}_target", forbidden)
        materialization = DisjointPatchRecombiner().materialize(
            ancestor=hypothesis.configuration,
            ancestor_candidate_id=hypothesis.candidate_id,
            left=model_endpoint.configuration,
            left_candidate_id=left_id,
            right=mate_endpoint.configuration,
            right_candidate_id=right_id,
            target_candidate_id=target_id,
        )
        materialization.revalidate()
        left_paths = tuple(
            sorted(
                _path_text(operation.path)
                for operation in materialization.classification.left_patch.operations
            )
        )
        right_paths = tuple(
            sorted(
                _path_text(operation.path)
                for operation in materialization.classification.right_patch.operations
            )
        )
        if (
            left_paths != model_endpoint.changed_paths
            or right_paths != mate_endpoint.changed_paths
        ):
            raise ValueError("prospective union did not bind complete branch support")
        configuration_sha256 = typed_json_sha256(materialization.configuration)
        phenotype = self.engine.identify_phenotype(materialization.configuration)
        return (
            _ProspectiveUnion(
                slot_id=slot_id,
                configuration=materialization.configuration,
                configuration_sha256=configuration_sha256,
                phenotype_identity_sha256=phenotype.identity_sha256,
                prospective_receipt_sha256=materialization.receipt_sha256,
            ),
            materialization,
        )

    def _g2(self, state: OptimizerState) -> GenerationPlan:
        if self.wave is None or self.genesis is None:
            raise RuntimeError("G1 diagnostic wave is unavailable")
        self._require_exact_state(state)
        if not self._g1_expected or not self._runtime_hypothesis_matrix:
            raise RuntimeError("G1 prospective/runtime authorities are unavailable")
        g1_receipt = state.generation_receipts[0]
        if tuple(value.slot.slot_id for value in g1_receipt.slot_results) != (
            G1_DIAGNOSTIC_SLOT_IDS
        ):
            raise ValueError("G1 receipt differs from the frozen slot order")
        expected_by_ref = {value.reference: value for value in self._g1_expected}
        g1_children: list[EvolutionCandidate] = []
        for result in g1_receipt.slot_results:
            assignment = result.outcome.prepared.plan.resolved_insight_assignment
            if assignment is None or assignment.arm is not MemoryAssignmentArm.DIAGNOSTIC:
                raise ValueError("G1 outcome lost its diagnostic assignment")
            reference = assignment.selection_decision.selected
            if len(reference) != 1 or reference[0] not in expected_by_ref:
                raise ValueError("G1 outcome selected a foreign hypothesis")
            g1_children.append(
                self._require_model_endpoint(
                    result.outcome,
                    expected=expected_by_ref[reference[0]],
                    assignment_role=TreatmentAssignmentRole.ACTIVE,
                    assignment_kind=InsightAssignmentKind.RESOLVED_CAUSAL,
                    generation=1,
                )
            )
        if len({self._phenotype_sha256(value) for value in g1_children}) != 2:
            raise ValueError("G1 actual hypothesis phenotypes collided")
        self.g1_rendered_prompt_receipt = self._prompt_receipt(
            g1_receipt,
            G1_DIAGNOSTIC_SLOT_IDS,
        )
        self.closure = self.checkpoint_service.close_generation(
            self.wave,
            g1_receipt,
        )
        if self.closure.status is not MemoryCheckpointClosureStatus.SEALED:
            raise RuntimeError("G1 causal memory wave did not seal")
        snapshot = self.closure.snapshot
        if snapshot is None:
            raise RuntimeError("sealed G1 wave has no score checkpoint")
        if any(not entry.identified for entry in snapshot.entries):
            raise ValueError("G1 did not identify both active hypothesis effects")
        scores = tuple(entry.retrieval_score for entry in snapshot.entries)
        if scores[0] == scores[1]:
            raise ValueError("G1 active hypothesis scores tied")

        _, hypothesis = self._parents(state)
        contract = self.benchmark.bind_finite_variation(
            self.model_catalog_id,
            hypothesis.configuration,
        )
        matrix = self._runtime_hypothesis_matrix
        if any(
            value.request.finite_contract.identity_sha256 != contract.identity_sha256
            for value in matrix
        ):
            raise ValueError("frozen P_H compilation differs from runtime catalog")
        base = self._base_model_plan(
            parent=hypothesis,
            generation=2,
            label="g2_model",
            contract=contract,
        )
        self.g2_prompt_shape_sha256 = self.engine.prompt_shape_commitment(
            base,
            selected_insight_count=1,
            reward_definition_hash=self.reward_binding.definition_hash,
        )
        assignments = (
            ResolvedInsightAssignment.resolve(
                credit_unit_id=self.ids.new_operator_invocation_id(),
                snapshot=snapshot,
                expected_snapshot_sha256=snapshot.snapshot_sha256,
                block_id="g2_matched_block",
                arm=MemoryAssignmentArm.ADAPTIVE,
                selection_decision=self.controls.adaptive(
                    snapshot=snapshot,
                    subset_size=1,
                ),
                prompt_shape_sha256=self.g2_prompt_shape_sha256,
            ),
            ResolvedInsightAssignment.resolve(
                credit_unit_id=self.ids.new_operator_invocation_id(),
                snapshot=snapshot,
                expected_snapshot_sha256=snapshot.snapshot_sha256,
                block_id="g2_matched_block",
                arm=MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
                selection_decision=self.controls.score_shuffled(
                    snapshot=snapshot,
                    subset_size=1,
                    permutation_rank=1,
                ),
                prompt_shape_sha256=self.g2_prompt_shape_sha256,
            ),
        )
        if assignments[0].selection_decision.selected == (
            assignments[1].selection_decision.selected
        ):
            raise ValueError("score-shuffled G2 control did not derange selection")
        self.g2_assignments = assignments
        by_ref = {value.request.reference: value for value in matrix}
        active_slots = tuple(
            OptimizerSlot.model(
                slot_id=G2_SLOT_IDS[index],
                role=("adaptive_active" if index == 0 else "score_shuffled_active"),
                plan=replace(
                    base,
                    label=G2_SLOT_IDS[index],
                    resolved_insight_assignment=assignment,
                    insight_treatment_requirement=(
                        by_ref[assignment.selection_decision.selected[0]].requirement
                    ),
                    compiled_hypothesis_treatment=(
                        by_ref[assignment.selection_decision.selected[0]]
                    ),
                    compiled_hypothesis_eligibility=matrix,
                ),
            )
            for index, assignment in enumerate(assignments)
        )

        neutral_entry = self.memory.entries_for((self.neutral_reference,))[0]
        neutral_requirement = _neutral_sham_requirement(
            entry=neutral_entry,
            contract=contract,
            choice=self.neutral_choice,
        )
        neutral_plan = replace(
            base,
            label=G2_SLOT_IDS[2],
            quarantine_test_insights=(neutral_entry.reference,),
            insight_treatment_requirement=neutral_requirement,
        )
        neutral_shape = self.engine.prompt_shape_commitment(
            neutral_plan,
            selected_insight_count=1,
            reward_definition_hash=self.reward_binding.definition_hash,
        )
        if neutral_shape != self.g2_prompt_shape_sha256:
            raise ValueError("G2 sham prompt-shape commitment is unmatched")

        mate_contract = self.benchmark.bind_finite_variation(
            self.mate_choice.catalog_id,
            hypothesis.configuration,
        )
        mate = _materialized_finite_choice(
            ids=self.ids,
            parent=hypothesis,
            generation=2,
            label=G2_SLOT_IDS[3],
            contract=mate_contract,
            choice=self.mate_choice,
        )
        active_expected = tuple(
            self._endpoint(
                slot_id=G2_SLOT_IDS[index],
                reference=assignment.selection_decision.selected[0],
                parent=hypothesis,
                contract=contract,
                option_id=by_ref[
                    assignment.selection_decision.selected[0]
                ].requirement.allowed_actions[0].option_id,
            )
            for index, assignment in enumerate(assignments)
        )
        neutral_expected = self._endpoint(
            slot_id=G2_SLOT_IDS[2],
            reference=neutral_entry.reference,
            parent=hypothesis,
            contract=contract,
            option_id=self.neutral_choice.option_id,
        )
        mate_expected = self._endpoint(
            slot_id=G2_SLOT_IDS[3],
            reference=None,
            parent=hypothesis,
            contract=mate_contract,
            option_id=self.mate_choice.option_id,
        )
        if (
            typed_json_sha256(freeze_json(mate.draft.configuration))
            != mate_expected.configuration_sha256
        ):
            raise RuntimeError("engine mate materialization differs from frozen choice")
        model_endpoints = (*active_expected, neutral_expected)
        if len({value.option_identity_sha256 for value in model_endpoints}) != 3:
            raise ValueError("G2 A/S/N actions are not pairwise distinct")
        if len({value.phenotype_identity_sha256 for value in model_endpoints}) != 3:
            raise ValueError("G2 A/S/N treatments do not produce three phenotypes")

        prospective_pairs = tuple(
            self._prospective_union(
                hypothesis=hypothesis,
                model_endpoint=endpoint,
                mate_endpoint=mate_expected,
                slot_id=slot_id,
            )
            for endpoint, slot_id in zip(
                model_endpoints,
                G3_SLOT_IDS[1:],
                strict=True,
            )
        )
        prospective_unions = tuple(value[0] for value in prospective_pairs)
        historical_phenotypes = {
            self._phenotype_sha256(candidate) for candidate in state.candidates
        }
        all_new_phenotypes = tuple(
            value.phenotype_identity_sha256
            for value in (*model_endpoints, mate_expected, *prospective_unions)
        )
        if len(set(all_new_phenotypes)) != 7 or historical_phenotypes.intersection(
            all_new_phenotypes
        ):
            raise ValueError(
                "prospective G2/G3 endpoints do not prove seven fresh phenotypes"
            )
        self._g2_expected = (*model_endpoints, mate_expected)
        self._g2_prospective_unions = prospective_unions
        self._g2_prospective_proof_sha256 = _hash(
            _PROSPECTIVE_DOMAIN,
            {
                "historical_phenotype_sha256s": sorted(historical_phenotypes),
                "endpoints": [
                    {
                        "slot_id": value.slot_id,
                        "configuration_sha256": value.configuration_sha256,
                        "phenotype_identity_sha256": (
                            value.phenotype_identity_sha256
                        ),
                        "changed_paths": list(value.changed_paths),
                    }
                    for value in self._g2_expected
                ],
                "unions": [
                    {
                        "slot_id": value.slot_id,
                        "configuration_sha256": value.configuration_sha256,
                        "phenotype_identity_sha256": (
                            value.phenotype_identity_sha256
                        ),
                        "prospective_receipt_sha256": (
                            value.prospective_receipt_sha256
                        ),
                    }
                    for value in prospective_unions
                ],
            },
        )

        return GenerationPlan(
            generation=2,
            slots=(
                *active_slots,
                OptimizerSlot.model(
                    slot_id=G2_SLOT_IDS[2],
                    role="evidence_free_sham_control",
                    plan=neutral_plan,
                ),
                OptimizerSlot.engine(
                    slot_id=G2_SLOT_IDS[3],
                    role="orthogonal_engine_mate",
                    invocation=mate,
                ),
            ),
            reward=self._reward(state, 2),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("g1_rendered_prompt_receipt_sha256", self.g1_rendered_prompt_receipt.receipt_sha256),
                        ("mate_choice_sha256", self.mate_choice.choice_sha256),
                        ("memory_snapshot_sha256", snapshot.snapshot_sha256),
                        ("neutral_choice_sha256", self.neutral_choice.choice_sha256),
                        ("prompt_shape_sha256", self.g2_prompt_shape_sha256),
                        ("prospective_g2_g3_proof_sha256", self._g2_prospective_proof_sha256),
                    )
                )
            ),
        )

    def _g3(self, state: OptimizerState) -> GenerationPlan:
        self._require_exact_state(state)
        if len(self._g2_expected) != 4 or len(self._g2_prospective_unions) != 3:
            raise RuntimeError("G2 prospective authority is unavailable")
        _, hypothesis = self._parents(state)
        g2 = state.generation_receipts[1]
        if tuple(value.slot.slot_id for value in g2.slot_results) != G2_SLOT_IDS:
            raise ValueError("G2 receipt slot order differs from frozen contract")
        active_children = tuple(
            self._require_model_endpoint(
                result.outcome,
                expected=expected,
                assignment_role=TreatmentAssignmentRole.ACTIVE,
                assignment_kind=InsightAssignmentKind.RESOLVED_CAUSAL,
                generation=2,
            )
            for result, expected in zip(
                g2.slot_results[:2],
                self._g2_expected[:2],
                strict=True,
            )
        )
        sham = self._require_model_endpoint(
            g2.slot_results[2].outcome,
            expected=self._g2_expected[2],
            assignment_role=TreatmentAssignmentRole.SHAM_CONTROL,
            assignment_kind=InsightAssignmentKind.QUARANTINE_TEST,
            generation=2,
        )
        mate = self._require_engine_endpoint(
            g2.slot_results[3].outcome,
            expected=self._g2_expected[3],
            generation=2,
        )
        adaptive, shuffled = active_children
        actual_g2_phenotypes = tuple(
            self._phenotype_sha256(value)
            for value in (adaptive, shuffled, sham, mate)
        )
        if len(set(actual_g2_phenotypes)) != 4:
            raise ValueError("actual G2 A/S/N/E phenotypes collided")
        self.g2_rendered_prompt_receipt = self._prompt_receipt(
            g2,
            G2_SLOT_IDS[:3],
        )

        reproduction = InvocationPlan(
            operator_kind=OperatorKind.REPRODUCTION,
            parents=(hypothesis,),
            generation=3,
            label=G3_SLOT_IDS[0],
            phase="g3_reproduction_control",
        )

        def union(
            model_child: EvolutionCandidate,
            slot_id: str,
        ) -> MaterializedInvocation:
            materialization = DisjointPatchRecombiner().materialize(
                ancestor=hypothesis.configuration,
                ancestor_candidate_id=hypothesis.candidate_id,
                left=model_child.configuration,
                left_candidate_id=model_child.candidate_id,
                right=mate.configuration,
                right_candidate_id=mate.candidate_id,
                target_candidate_id=self.ids.new_candidate_id(),
            )
            plan = InvocationPlan(
                operator_kind=OperatorKind.THREE_WAY_RECOMBINATION,
                parents=(model_child, mate),
                generation=3,
                label=slot_id,
                common_ancestor=hypothesis,
                phase="g3_disjoint_union",
            )
            return materialized_disjoint_invocation(
                plan=plan,
                materialization=materialization,
            )

        unions = tuple(
            union(child, slot_id)
            for child, slot_id in zip(
                (adaptive, shuffled, sham),
                G3_SLOT_IDS[1:],
                strict=True,
            )
        )
        for invocation, expected in zip(
            unions,
            self._g2_prospective_unions,
            strict=True,
        ):
            observed_configuration = freeze_json(invocation.draft.configuration)
            if (
                typed_json_sha256(observed_configuration)
                != expected.configuration_sha256
                or not typed_json_equal(
                    observed_configuration,
                    expected.configuration,
                )
                or self.engine.identify_phenotype(observed_configuration).identity_sha256
                != expected.phenotype_identity_sha256
            ):
                raise ValueError(
                    "actual G3 union differs from prospective disjoint replay"
                )
        slots = (
            OptimizerSlot.reproduction(
                slot_id=G3_SLOT_IDS[0],
                role="hypothesis_parent_reproduction",
                plan=reproduction,
            ),
            *(
                OptimizerSlot.engine(
                    slot_id=slot_id,
                    role="deterministic_disjoint_union",
                    invocation=invocation,
                )
                for slot_id, invocation in zip(
                    G3_SLOT_IDS[1:], unions, strict=True
                )
            ),
        )
        if any(
            slot.proposal_authority is ProposalAuthority.MODEL for slot in slots
        ):
            raise RuntimeError("G3 must contain zero model calls")
        if (
            self._seed_occurrence_binding_sha256 is None
            or self._seed_phenotype_sha256s is None
            or self._g2_prospective_proof_sha256 is None
            or self.genesis is None
            or self.wave is None
            or self.closure is None
            or self.closure.snapshot is None
            or self.g1_rendered_prompt_receipt is None
            or self.g2_rendered_prompt_receipt is None
        ):
            raise RuntimeError("G3 terminal authority prerequisites are unavailable")

        def endpoint_authority(
            value: _ProspectiveEndpoint,
        ) -> G3ExpectedEndpoint:
            return G3ExpectedEndpoint(
                slot_id=value.slot_id,
                reference=value.reference,
                option_id=value.option_id,
                option_identity_sha256=value.option_identity_sha256,
                configuration=value.configuration,
                configuration_sha256=value.configuration_sha256,
                phenotype_identity_sha256=value.phenotype_identity_sha256,
                changed_paths=value.changed_paths,
            )

        g1_expected_by_reference = {
            value.reference: value for value in self._g1_expected
        }
        # The frozen wave canonicalizes assignments by receipt hash.  Recover
        # the actual slot realization from the G1 receipt instead of relying on
        # that storage order when the public permutation rank is non-zero.
        g1_receipt = state.generation_receipts[0]

        def realized_g1_endpoint(result) -> _ProspectiveEndpoint:
            assignment = result.outcome.prepared.plan.resolved_insight_assignment
            if assignment is None or len(assignment.selection_decision.selected) != 1:
                raise RuntimeError("G1 terminal authority lost its assignment")
            return replace(
                g1_expected_by_reference[
                    assignment.selection_decision.selected[0]
                ],
                slot_id=result.slot.slot_id,
            )

        realized_g1_expected = tuple(
            realized_g1_endpoint(result) for result in g1_receipt.slot_results
        )
        terminal_authority = G3TerminalValidationAuthority(
            hypothesis_parent_candidate_id=hypothesis.candidate_id,
            hypothesis_parent_configuration=hypothesis.configuration,
            hypothesis_parent_configuration_sha256=(
                hypothesis.occurrence.configuration_hash
            ),
            hypothesis_parent_phenotype_identity_sha256=(
                self._phenotype_sha256(hypothesis)
            ),
            seed_occurrence_binding_sha256=self._seed_occurrence_binding_sha256,
            seed_phenotype_identity_sha256s=self._seed_phenotype_sha256s,
            g1_expected_endpoints=tuple(
                endpoint_authority(value) for value in realized_g1_expected
            ),
            g2_expected_endpoints=tuple(
                endpoint_authority(value) for value in self._g2_expected
            ),
            g3_expected_unions=tuple(
                G3ExpectedUnion(
                    slot_id=expected.slot_id,
                    configuration=expected.configuration,
                    configuration_sha256=expected.configuration_sha256,
                    phenotype_identity_sha256=(
                        expected.phenotype_identity_sha256
                    ),
                    prospective_materialization_receipt_sha256=(
                        expected.prospective_receipt_sha256
                    ),
                    runtime_materialization_receipt_sha256=(
                        invocation.materialization_receipt_hash
                    ),
                )
                for expected, invocation in zip(
                    self._g2_prospective_unions,
                    unions,
                    strict=True,
                )
            ),
            prospective_proof_sha256=self._g2_prospective_proof_sha256,
            g1_rendered_prompt_receipt_sha256=(
                self.g1_rendered_prompt_receipt.receipt_sha256
            ),
            g2_rendered_prompt_receipt_sha256=(
                self.g2_rendered_prompt_receipt.receipt_sha256
            ),
            genesis_snapshot_sha256=self.genesis.snapshot_sha256,
            diagnostic_wave_sha256=self.wave.wave_sha256,
            closure_snapshot_sha256=self.closure.snapshot.snapshot_sha256,
        )
        if self._terminal_validation_authority is not None:
            if (
                self._terminal_validation_authority.authority_sha256
                != terminal_authority.authority_sha256
            ):
                raise RuntimeError("G3 terminal authority changed after freezing")
        else:
            self._terminal_validation_authority = terminal_authority
        return GenerationPlan(
            generation=3,
            slots=slots,
            reward=self._reward(state, 3),
            planner_policy_id=self.policy_id,
            planner_policy_version=self.policy_version,
            metadata=tuple(
                sorted(
                    (
                        ("g2_rendered_prompt_receipt_sha256", self.g2_rendered_prompt_receipt.receipt_sha256),
                        ("prospective_g2_g3_proof_sha256", self._g2_prospective_proof_sha256),
                        (
                            "terminal_validation_authority_sha256",
                            terminal_authority.authority_sha256,
                        ),
                        *(
                            (
                                f"{slot_id}_receipt_sha256",
                                invocation.materialization_receipt_hash,
                            )
                            for slot_id, invocation in zip(
                                G3_SLOT_IDS[1:],
                                unions,
                                strict=True,
                            )
                        ),
                    )
                )
            ),
        )


__all__ = [
    "G1_DIAGNOSTIC_SLOT_IDS",
    "G2_SLOT_IDS",
    "G3_SLOT_IDS",
    "G3BenchmarkBoundary",
    "G3CausalScreenPlanner",
    "G3ExpectedEndpoint",
    "G3ExpectedUnion",
    "G3_SCREEN_BUDGET",
    "G3_SCREEN_POLICY_ID",
    "G3_SCREEN_POLICY_VERSION",
    "FrozenDiagnosticPermutation",
    "G3TerminalValidationAuthority",
    "ParentBoundActionChoice",
    "PreparedHypothesisMatrix",
    "finite_mutation_boundary",
]
