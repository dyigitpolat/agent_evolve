"""Application service for evidence-bound executable hypothesis treatments.

The compiler maps an immutable, domain-neutral insight into exact options from
one parent-bound benchmark catalog.  This module turns a validated compilation
into the existing strict treatment-preflight vocabulary without losing the
source evidence, compiler receipt, or executable-spec identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.insight_memory import InsightMemoryEntry
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.treatment_compliance import (
    InsightTreatmentRequirement,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentInsightEvidence,
)
from agent_evolve.ports.executable_hypothesis import (
    HypothesisApplicabilityPort,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    validate_hypothesis_compiler_identity,
    validate_hypothesis_compilation,
)


_SOURCE_EVIDENCE_DOMAIN = b"agent-evolve:registered-insight-evidence:v1\x00"
_COMPILED_TREATMENT_DOMAIN = b"agent-evolve:compiled-hypothesis-treatment:v1\x00"
_COMPILER_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")


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


def registered_source_evidence_record(
    entry: InsightMemoryEntry,
) -> dict[str, object]:
    """Project every immutable evidence-bearing field of a registered card."""

    if type(entry) is not InsightMemoryEntry:
        raise TypeError("entry must be an exact InsightMemoryEntry")
    InsightMemoryEntry.__post_init__(entry)
    if type(entry.initial_score) is not float or not math.isfinite(entry.initial_score):
        raise TypeError("entry initial_score must be a finite canonical float")
    return {
        "schema_version": 1,
        "reference": {
            "insight_id": entry.reference.insight_id.value,
            "version": entry.reference.version,
        },
        "insight_content_sha256": entry.draft.content_sha256,
        "initial_score_hex": entry.initial_score.hex(),
        "applicable_operator_kinds": list(entry.applicable_operator_kinds),
        "lifecycle_state": entry.lifecycle_state.value,
        "origin": entry.origin.value,
        "evidence_lineage": (
            None
            if entry.evidence_lineage is None
            else entry.evidence_lineage.to_record()
        ),
        "relations": [
            {
                "kind": relation.kind.value,
                "target": {
                    "insight_id": relation.target.insight_id.value,
                    "version": relation.target.version,
                },
                "note": relation.note,
            }
            for relation in entry.relations
        ],
    }


def registered_source_evidence_sha256(entry: InsightMemoryEntry) -> str:
    """Hash the complete registered source projection before compilation."""

    return _hash(_SOURCE_EVIDENCE_DOMAIN, registered_source_evidence_record(entry))


@dataclass(frozen=True, slots=True)
class CompiledHypothesisTreatment:
    """Parallel v1 requirement that binds source, compiler, and strict treatment.

    ``InsightTreatmentRequirement`` remains unchanged for historical workflows.
    This enclosing value is the plan-hashed authority for compiled treatments;
    the engine must never reconstruct its evidence from raw card recommendations.
    """

    request: HypothesisCompilationRequest
    receipt: HypothesisCompilationReceipt
    treatment_evidence: TreatmentInsightEvidence
    requirement: InsightTreatmentRequirement
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.request) is not HypothesisCompilationRequest:
            raise TypeError("request must be an exact HypothesisCompilationRequest")
        if type(self.receipt) is not HypothesisCompilationReceipt:
            raise TypeError("receipt must be an exact HypothesisCompilationReceipt")
        validate_hypothesis_compilation(self.request, self.receipt)
        if not self.receipt.applicable or self.receipt.spec is None:
            raise ValueError("compiled treatment requires an applicable receipt")
        spec = self.receipt.spec
        if type(self.treatment_evidence) is not TreatmentInsightEvidence:
            raise TypeError(
                "treatment_evidence must be exact TreatmentInsightEvidence"
            )
        TreatmentInsightEvidence.__post_init__(self.treatment_evidence)
        expected_evidence = (
            spec.reference,
            spec.insight_content_sha256,
            spec.executable_operator_kinds,
            spec.affected_paths,
            spec.recommended_option_families,
            tuple(action.option_id for action in spec.allowed_actions),
        )
        observed_evidence = (
            self.treatment_evidence.reference,
            self.treatment_evidence.insight_content_sha256,
            self.treatment_evidence.applicable_operator_kinds,
            self.treatment_evidence.affected_paths,
            self.treatment_evidence.recommended_option_families,
            self.treatment_evidence.recommended_option_ids,
        )
        if observed_evidence != expected_evidence:
            raise ValueError(
                "compiled treatment evidence differs from executable spec"
            )
        if type(self.requirement) is not InsightTreatmentRequirement:
            raise TypeError("requirement must be an InsightTreatmentRequirement")
        InsightTreatmentRequirement.__post_init__(self.requirement)
        expected_requirement = (
            (self.treatment_evidence.binding(),),
            spec.finite_contract_sha256,
            spec.allowed_actions,
            TreatmentClaimMode.EXACT_REQUIRED,
            TreatmentAssignmentRole.ACTIVE,
            True,
            True,
        )
        observed_requirement = (
            self.requirement.insight_bindings,
            self.requirement.finite_contract_sha256,
            self.requirement.allowed_actions,
            self.requirement.claim_mode,
            self.requirement.assignment_role,
            self.requirement.require_option_family_match,
            self.requirement.require_changed_path_overlap,
        )
        if observed_requirement != expected_requirement:
            raise ValueError(
                "compiled treatment requirement differs from executable spec"
            )
        object.__setattr__(
            self,
            "binding_sha256",
            _hash(_COMPILED_TREATMENT_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        spec = self.receipt.spec
        return {
            "schema_version": 1,
            "request_sha256": self.request.request_sha256,
            "source_evidence_sha256": self.request.source_evidence_sha256,
            "compiler_receipt_sha256": self.receipt.receipt_sha256,
            "executable_spec_sha256": (
                None if spec is None else spec.spec_sha256
            ),
            "treatment_evidence": {
                **self.treatment_evidence.to_record(),
                "evidence_sha256": self.treatment_evidence.evidence_sha256,
            },
            "requirement": {
                **self.requirement.to_record(),
                "requirement_sha256": self.requirement.requirement_sha256,
            },
        }


def _validate_registered_request(
    *,
    entry: InsightMemoryEntry,
    request: HypothesisCompilationRequest,
) -> None:
    if type(entry) is not InsightMemoryEntry:
        raise TypeError("entry must be an exact InsightMemoryEntry")
    InsightMemoryEntry.__post_init__(entry)
    if type(request) is not HypothesisCompilationRequest:
        raise TypeError("request must be an exact HypothesisCompilationRequest")
    request.__post_init__()
    expected_source = registered_source_evidence_sha256(entry)
    expected_entry = (
        entry.reference,
        entry.draft.content_sha256,
        entry.applicable_operator_kinds,
        expected_source,
    )
    observed_entry = (
        request.reference,
        request.insight.content_sha256,
        request.source_operator_kinds,
        request.source_evidence_sha256,
    )
    if observed_entry != expected_entry:
        raise ValueError(
            "hypothesis compilation request differs from registered source evidence"
        )


def _join_compilation_receipt(
    *,
    request: HypothesisCompilationRequest,
    receipt: HypothesisCompilationReceipt,
) -> CompiledHypothesisTreatment:
    """Private join used only after the injected compiler has been authenticated."""

    if not receipt.applicable or receipt.spec is None:
        raise ValueError("cannot build a treatment from an inapplicable compilation")
    spec = receipt.spec
    treatment_evidence = TreatmentInsightEvidence(
        reference=spec.reference,
        insight_content_sha256=spec.insight_content_sha256,
        applicable_operator_kinds=spec.executable_operator_kinds,
        affected_paths=spec.affected_paths,
        recommended_option_families=spec.recommended_option_families,
        recommended_option_ids=tuple(
            action.option_id for action in spec.allowed_actions
        ),
    )
    requirement = InsightTreatmentRequirement(
        insight_bindings=(treatment_evidence.binding(),),
        finite_contract_sha256=spec.finite_contract_sha256,
        allowed_actions=spec.allowed_actions,
        claim_mode=TreatmentClaimMode.EXACT_REQUIRED,
        assignment_role=TreatmentAssignmentRole.ACTIVE,
        require_option_family_match=True,
        require_changed_path_overlap=True,
    )
    return CompiledHypothesisTreatment(
        request=request,
        receipt=receipt,
        treatment_evidence=treatment_evidence,
        requirement=requirement,
    )


def compile_registered_hypothesis_treatment(
    *,
    entry: InsightMemoryEntry,
    request: HypothesisCompilationRequest,
    compiler: HypothesisApplicabilityPort,
) -> CompiledHypothesisTreatment:
    """Own the compiler call and close identity/result injection seams."""

    _validate_registered_request(entry=entry, request=request)
    frozen_request_sha256 = request.request_sha256
    frozen_source_sha256 = registered_source_evidence_sha256(entry)
    if not isinstance(compiler, HypothesisApplicabilityPort):
        raise TypeError("compiler must implement HypothesisApplicabilityPort")
    frozen_identity = (
        compiler.policy_id,
        compiler.policy_version,
        compiler.definition_sha256,
    )
    if (
        type(frozen_identity[0]) is not str
        or _COMPILER_ID.fullmatch(frozen_identity[0]) is None
    ):
        raise ValueError("compiler policy_id must use the canonical token grammar")
    if type(frozen_identity[1]) is not int or frozen_identity[1] <= 0:
        raise ValueError("compiler policy_version must be positive")
    require_sha256(frozen_identity[2], "compiler definition_sha256")
    receipt = compiler.compile(request)
    if request.request_sha256 != frozen_request_sha256:
        raise ValueError("hypothesis compilation request changed during compilation")
    if registered_source_evidence_sha256(entry) != frozen_source_sha256:
        raise ValueError("registered source evidence changed during compilation")
    _validate_registered_request(entry=entry, request=request)
    current_identity = (
        compiler.policy_id,
        compiler.policy_version,
        compiler.definition_sha256,
    )
    receipt_identity = (
        receipt.compiler_policy_id,
        receipt.compiler_policy_version,
        receipt.compiler_definition_sha256,
    )
    if current_identity != frozen_identity:
        raise ValueError("compiler identity changed during compilation")
    if receipt_identity != frozen_identity:
        raise ValueError("compiler receipt differs from frozen compiler identity")
    validate_hypothesis_compiler_identity(compiler, receipt)
    validate_hypothesis_compilation(request, receipt)
    return _join_compilation_receipt(request=request, receipt=receipt)


__all__ = [
    "CompiledHypothesisTreatment",
    "compile_registered_hypothesis_treatment",
    "registered_source_evidence_record",
    "registered_source_evidence_sha256",
]
