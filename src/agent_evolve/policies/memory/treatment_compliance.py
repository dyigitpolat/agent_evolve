"""Generic, evaluator-free administration of assigned insight treatments.

The policy operates only on immutable machine evidence: exact insight versions,
finite-option families, and parent-relative changed paths.  It never inspects an
objective or benchmark-specific field.  A rejected model proposal therefore can
be recorded as no-yield before an expensive evaluator is entered.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_POLICY_DEFINITION = (
    b"agent-evolve:treatment-compliance:v2:exact-card-palette-action-claims-before-eval"
)
TREATMENT_COMPLIANCE_POLICY_ID = "strict_treatment_compliance"
TREATMENT_COMPLIANCE_POLICY_VERSION = 2
TREATMENT_COMPLIANCE_DEFINITION_SHA256 = hashlib.sha256(
    _POLICY_DEFINITION
).hexdigest()


def _hash(
    domain: str,
    record: dict[str, object],
    *,
    version: int = 1,
) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        f"agent-evolve:{domain}:v{version}\x00".encode("ascii") + payload
    ).hexdigest()


def _ref_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _canonical_strings(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or not value for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of non-empty strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


def _overlaps_any(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return any(_paths_overlap(left, right) for left in first for right in second)


class TreatmentClaimMode(str, Enum):
    """How an assigned treatment must be reported by the model."""

    OPTIONAL_SUBSET = "optional_subset"
    EXACT_REQUIRED = "exact_required"


class TreatmentAssignmentRole(str, Enum):
    """Scientific role of the assigned card, independent of benchmark names."""

    ACTIVE = "active"
    SHAM_CONTROL = "sham_control"


@dataclass(frozen=True, slots=True)
class TreatmentActionBinding:
    """One exact parent-bound finite action permitted by a treatment."""

    option_id: str
    option_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _TOKEN.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the canonical option grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")

    def to_record(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class TreatmentInsightBinding:
    """Plan-hashed identity of one exact insight card and its action claim."""

    reference: InsightRef
    insight_content_sha256: str
    evidence_sha256: str
    recommended_option_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.insight_content_sha256, "insight_content_sha256")
        require_sha256(self.evidence_sha256, "evidence_sha256")
        _canonical_strings(
            self.recommended_option_ids,
            name="recommended_option_ids",
        )
        if any(_TOKEN.fullmatch(value) is None for value in self.recommended_option_ids):
            raise ValueError("recommended_option_ids use the canonical option grammar")

    def to_record(self) -> dict[str, object]:
        return {
            **_ref_record(self.reference),
            "insight_content_sha256": self.insight_content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "recommended_option_ids": list(self.recommended_option_ids),
        }


@dataclass(frozen=True, slots=True)
class InsightTreatmentRequirement:
    """Plan-hashed exact card, palette, and parent-bound action contract."""

    insight_bindings: tuple[TreatmentInsightBinding, ...]
    finite_contract_sha256: str
    allowed_actions: tuple[TreatmentActionBinding, ...]
    claim_mode: TreatmentClaimMode
    assignment_role: TreatmentAssignmentRole = TreatmentAssignmentRole.ACTIVE
    require_option_family_match: bool = True
    require_changed_path_overlap: bool = True
    requirement_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.insight_bindings) is not tuple or not self.insight_bindings:
            raise ValueError("a treatment requirement needs assigned insights")
        if any(
            type(value) is not TreatmentInsightBinding
            for value in self.insight_bindings
        ):
            raise TypeError(
                "insight_bindings must contain exact TreatmentInsightBinding values"
            )
        for binding in self.insight_bindings:
            TreatmentInsightBinding.__post_init__(binding)
        references = tuple(value.reference for value in self.insight_bindings)
        if references != tuple(sorted(set(references))):
            raise ValueError("insight_bindings must be unique and canonically sorted")
        ids = tuple(value.insight_id.value for value in references)
        if len(set(ids)) != len(ids):
            raise ValueError(
                "unversioned claimed_insight_ids require distinct logical insight IDs"
            )
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")
        if type(self.allowed_actions) is not tuple or not self.allowed_actions:
            raise ValueError("a treatment requirement needs exact allowed actions")
        if any(
            type(value) is not TreatmentActionBinding
            for value in self.allowed_actions
        ):
            raise TypeError(
                "allowed_actions must contain exact TreatmentActionBinding values"
            )
        for action in self.allowed_actions:
            TreatmentActionBinding.__post_init__(action)
        if self.allowed_actions != tuple(
            sorted(
                set(self.allowed_actions),
                key=lambda value: (value.option_id, value.option_identity_sha256),
            )
        ):
            raise ValueError("allowed_actions must be unique and canonically sorted")
        common_option_ids = set.intersection(
            *(set(binding.recommended_option_ids) for binding in self.insight_bindings)
        )
        allowed_option_ids = {action.option_id for action in self.allowed_actions}
        if allowed_option_ids != common_option_ids:
            raise ValueError(
                "allowed action IDs must equal the cards' exact common recommendation"
            )
        if len(allowed_option_ids) != len(self.allowed_actions):
            raise ValueError("allowed action option IDs must be unique")
        if type(self.claim_mode) is not TreatmentClaimMode:
            raise TypeError("claim_mode must be a TreatmentClaimMode")
        if type(self.assignment_role) is not TreatmentAssignmentRole:
            raise TypeError("assignment_role must be a TreatmentAssignmentRole")
        if type(self.require_option_family_match) is not bool or type(
            self.require_changed_path_overlap
        ) is not bool:
            raise TypeError("treatment requirement switches must be bool")
        object.__setattr__(
            self,
            "requirement_sha256",
            _hash(
                "insight-treatment-requirement",
                self.to_record(),
                version=2,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "insight_bindings": [
                binding.to_record() for binding in self.insight_bindings
            ],
            "finite_contract_sha256": self.finite_contract_sha256,
            "allowed_actions": [
                action.to_record() for action in self.allowed_actions
            ],
            "claim_mode": self.claim_mode.value,
            "assignment_role": self.assignment_role.value,
            "require_option_family_match": self.require_option_family_match,
            "require_changed_path_overlap": self.require_changed_path_overlap,
        }

    @property
    def required_insights(self) -> tuple[InsightRef, ...]:
        return tuple(binding.reference for binding in self.insight_bindings)


@dataclass(frozen=True, slots=True)
class TreatmentInsightEvidence:
    reference: InsightRef
    insight_content_sha256: str
    applicable_operator_kinds: tuple[str, ...]
    affected_paths: tuple[str, ...]
    recommended_option_families: tuple[str, ...]
    recommended_option_ids: tuple[str, ...]
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        require_sha256(self.insight_content_sha256, "insight_content_sha256")
        for name in (
            "applicable_operator_kinds",
            "affected_paths",
            "recommended_option_families",
            "recommended_option_ids",
        ):
            _canonical_strings(getattr(self, name), name=name)
        if any(_TOKEN.fullmatch(value) is None for value in self.recommended_option_ids):
            raise ValueError("recommended_option_ids use the canonical option grammar")
        object.__setattr__(
            self,
            "evidence_sha256",
            _hash(
                "treatment-insight-evidence",
                self.to_record(),
                version=2,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            **_ref_record(self.reference),
            "insight_content_sha256": self.insight_content_sha256,
            "applicable_operator_kinds": list(self.applicable_operator_kinds),
            "affected_paths": list(self.affected_paths),
            "recommended_option_families": list(
                self.recommended_option_families
            ),
            "recommended_option_ids": list(self.recommended_option_ids),
        }

    def binding(self) -> TreatmentInsightBinding:
        return TreatmentInsightBinding(
            reference=self.reference,
            insight_content_sha256=self.insight_content_sha256,
            evidence_sha256=self.evidence_sha256,
            recommended_option_ids=self.recommended_option_ids,
        )


@dataclass(frozen=True, slots=True)
class FiniteTreatmentAction:
    option_id: str
    option_identity_sha256: str
    family: str
    changed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _TOKEN.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the canonical option grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the canonical option grammar")
        _canonical_strings(self.changed_paths, name="changed_paths")

    def to_record(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "family": self.family,
            "changed_paths": list(self.changed_paths),
        }

    def binding(self) -> TreatmentActionBinding:
        return TreatmentActionBinding(
            option_id=self.option_id,
            option_identity_sha256=self.option_identity_sha256,
        )


class TreatmentComplianceViolation(str, Enum):
    ASSIGNMENT_MISMATCH = "assignment_mismatch"
    INSIGHT_EVIDENCE_MISMATCH = "insight_evidence_mismatch"
    FINITE_CONTRACT_MISMATCH = "finite_contract_mismatch"
    EXACT_ACTION_BINDING_MISMATCH = "exact_action_binding_mismatch"
    OPERATOR_INAPPLICABLE = "operator_inapplicable"
    EDITABLE_PATH_DISJOINT = "editable_path_disjoint"
    INTERVENTION_FAMILY_MISSING = "intervention_family_missing"
    NO_COMPATIBLE_FINITE_ACTION = "no_compatible_finite_action"
    OPERATOR_NONCOMPLIANT = "operator_noncompliant"
    DUPLICATE_CLAIM = "duplicate_claim"
    FOREIGN_CLAIM = "foreign_claim"
    EXACT_CLAIM_MISMATCH = "exact_claim_mismatch"
    SELECTED_ACTION_INCOMPATIBLE = "selected_action_incompatible"


@dataclass(frozen=True, slots=True)
class TreatmentPreflightRequest:
    requirement: InsightTreatmentRequirement
    operator_kind: str
    editable_paths: tuple[str, ...]
    insights: tuple[TreatmentInsightEvidence, ...]
    finite_contract_sha256: str
    actions: tuple[FiniteTreatmentAction, ...]

    def __post_init__(self) -> None:
        if type(self.requirement) is not InsightTreatmentRequirement:
            raise TypeError("requirement must be an InsightTreatmentRequirement")
        InsightTreatmentRequirement.__post_init__(self.requirement)
        if type(self.operator_kind) is not str or not self.operator_kind:
            raise ValueError("operator_kind must be non-empty")
        _canonical_strings(self.editable_paths, name="editable_paths")
        if type(self.insights) is not tuple or any(
            type(value) is not TreatmentInsightEvidence for value in self.insights
        ):
            raise TypeError("insights must contain TreatmentInsightEvidence values")
        if type(self.actions) is not tuple or not self.actions or any(
            type(value) is not FiniteTreatmentAction for value in self.actions
        ):
            raise TypeError("actions must contain finite treatment actions")
        for action in self.actions:
            FiniteTreatmentAction.__post_init__(action)
        action_bindings = tuple(action.binding() for action in self.actions)
        if len(set(action_bindings)) != len(action_bindings):
            raise ValueError("actions must use unique exact option bindings")
        if len({action.option_id for action in self.actions}) != len(self.actions):
            raise ValueError("actions must use unique option IDs")
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")


@dataclass(frozen=True, slots=True)
class TreatmentPreflightReceipt:
    requirement_sha256: str
    finite_contract_sha256: str
    compatible_actions: tuple[FiniteTreatmentAction, ...]
    compatible_families: tuple[str, ...]
    violations: tuple[TreatmentComplianceViolation, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.requirement_sha256, "requirement_sha256")
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")
        if type(self.compatible_actions) is not tuple or any(
            type(value) is not FiniteTreatmentAction
            for value in self.compatible_actions
        ):
            raise TypeError("compatible_actions must contain exact actions")
        _canonical_strings(self.compatible_families, name="compatible_families")
        if type(self.violations) is not tuple or any(
            type(value) is not TreatmentComplianceViolation
            for value in self.violations
        ):
            raise TypeError("violations must contain exact violation values")
        if self.violations != tuple(sorted(set(self.violations), key=str)):
            raise ValueError("violations must be unique and canonical")
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash("treatment-preflight-receipt", self.to_record()),
        )

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "requirement_sha256": self.requirement_sha256,
            "finite_contract_sha256": self.finite_contract_sha256,
            "compatible_actions": [
                value.to_record() for value in self.compatible_actions
            ],
            "compatible_families": list(self.compatible_families),
            "violations": [value.value for value in self.violations],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class TreatmentAdmissionRequest:
    requirement: InsightTreatmentRequirement
    preflight: TreatmentPreflightReceipt
    claimed_insight_ids: tuple[str, ...]
    selected_action: FiniteTreatmentAction
    operator_compliant: bool

    def __post_init__(self) -> None:
        if type(self.requirement) is not InsightTreatmentRequirement:
            raise TypeError("requirement must be exact")
        InsightTreatmentRequirement.__post_init__(self.requirement)
        if type(self.preflight) is not TreatmentPreflightReceipt:
            raise TypeError("preflight must be exact")
        TreatmentPreflightReceipt.__post_init__(self.preflight)
        if self.preflight.requirement_sha256 != self.requirement.requirement_sha256:
            raise ValueError("preflight is bound to a different requirement")
        if not self.preflight.passed:
            raise ValueError("admission requires a passing preflight receipt")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must contain only non-empty strings"
            )
        if type(self.selected_action) is not FiniteTreatmentAction:
            raise TypeError("selected_action must be exact")
        if type(self.operator_compliant) is not bool:
            raise TypeError("operator_compliant must be bool")


@dataclass(frozen=True, slots=True)
class TreatmentAdmissionReceipt:
    preflight_receipt_sha256: str
    claimed_insight_ids: tuple[str, ...]
    selected_action: FiniteTreatmentAction
    violations: tuple[TreatmentComplianceViolation, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    evaluator_entered: bool = False
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.preflight_receipt_sha256, "preflight_receipt_sha256")
        if type(self.claimed_insight_ids) is not tuple:
            raise TypeError("claimed_insight_ids must be a tuple")
        if type(self.selected_action) is not FiniteTreatmentAction:
            raise TypeError("selected_action must be exact")
        if type(self.violations) is not tuple or any(
            type(value) is not TreatmentComplianceViolation
            for value in self.violations
        ):
            raise TypeError("violations must contain exact values")
        if self.violations != tuple(sorted(set(self.violations), key=str)):
            raise ValueError("violations must be unique and canonical")
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if self.evaluator_entered is not False:
            raise ValueError(
                "treatment admission receipts are issued before evaluator entry"
            )
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash("treatment-admission-receipt", self.to_record()),
        )

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "claimed_insight_ids": list(self.claimed_insight_ids),
            "selected_action": self.selected_action.to_record(),
            "violations": [value.value for value in self.violations],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "evaluator_entered": self.evaluator_entered,
        }


class TreatmentComplianceRejected(ValueError):
    """Expected model-treatment mismatch; the evaluator must not run."""

    def __init__(self, receipt: TreatmentAdmissionReceipt) -> None:
        if type(receipt) is not TreatmentAdmissionReceipt or receipt.passed:
            raise ValueError("rejection requires a failed admission receipt")
        super().__init__("model proposal failed its assigned treatment contract")
        self.receipt = receipt


@runtime_checkable
class TreatmentCompliancePolicy(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def preflight(
        self, request: TreatmentPreflightRequest
    ) -> TreatmentPreflightReceipt: ...

    def assess(
        self, request: TreatmentAdmissionRequest
    ) -> TreatmentAdmissionReceipt: ...


@dataclass(frozen=True, slots=True)
class StrictTreatmentCompliancePolicy:
    """Machine-verifiable treatment fidelity over a shared finite palette."""

    policy_id: str = TREATMENT_COMPLIANCE_POLICY_ID
    policy_version: int = TREATMENT_COMPLIANCE_POLICY_VERSION
    definition_sha256: str = TREATMENT_COMPLIANCE_DEFINITION_SHA256

    def preflight(
        self, request: TreatmentPreflightRequest
    ) -> TreatmentPreflightReceipt:
        TreatmentPreflightRequest.__post_init__(request)
        violations: set[TreatmentComplianceViolation] = set()
        expected = request.requirement.required_insights
        observed = tuple(value.reference for value in request.insights)
        if observed != expected:
            violations.add(TreatmentComplianceViolation.ASSIGNMENT_MISMATCH)
        observed_bindings = tuple(value.binding() for value in request.insights)
        if observed_bindings != request.requirement.insight_bindings:
            violations.add(
                TreatmentComplianceViolation.INSIGHT_EVIDENCE_MISMATCH
            )
        if (
            request.finite_contract_sha256
            != request.requirement.finite_contract_sha256
        ):
            violations.add(TreatmentComplianceViolation.FINITE_CONTRACT_MISMATCH)
        trusted_action_bindings = {
            action.binding() for action in request.actions
        }
        if not set(request.requirement.allowed_actions).issubset(
            trusted_action_bindings
        ):
            violations.add(
                TreatmentComplianceViolation.EXACT_ACTION_BINDING_MISMATCH
            )
        for insight in request.insights:
            if (
                insight.applicable_operator_kinds
                and request.operator_kind not in insight.applicable_operator_kinds
            ):
                violations.add(TreatmentComplianceViolation.OPERATOR_INAPPLICABLE)
            if request.requirement.require_changed_path_overlap and not _overlaps_any(
                request.editable_paths, insight.affected_paths
            ):
                violations.add(TreatmentComplianceViolation.EDITABLE_PATH_DISJOINT)

        family_sets = [
            set(insight.recommended_option_families)
            for insight in request.insights
        ]
        if request.requirement.require_option_family_match and any(
            not values for values in family_sets
        ):
            violations.add(TreatmentComplianceViolation.INTERVENTION_FAMILY_MISSING)
        common_families = (
            set.intersection(*family_sets) if family_sets else set()
        )
        compatible = tuple(
            action
            for action in request.actions
            if action.binding() in request.requirement.allowed_actions
            and (
                not request.requirement.require_option_family_match
                or action.family in common_families
            )
            and (
                not request.requirement.require_changed_path_overlap
                or all(
                    _overlaps_any(action.changed_paths, insight.affected_paths)
                    for insight in request.insights
                )
            )
        )
        if not compatible:
            violations.add(TreatmentComplianceViolation.NO_COMPATIBLE_FINITE_ACTION)
        return TreatmentPreflightReceipt(
            requirement_sha256=request.requirement.requirement_sha256,
            finite_contract_sha256=request.finite_contract_sha256,
            compatible_actions=compatible,
            compatible_families=tuple(sorted({value.family for value in compatible})),
            violations=tuple(sorted(violations, key=str)),
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
        )

    def assess(
        self, request: TreatmentAdmissionRequest
    ) -> TreatmentAdmissionReceipt:
        TreatmentAdmissionRequest.__post_init__(request)
        violations: set[TreatmentComplianceViolation] = set()
        claims = request.claimed_insight_ids
        required_ids = tuple(
            value.insight_id.value for value in request.requirement.required_insights
        )
        if len(set(claims)) != len(claims):
            violations.add(TreatmentComplianceViolation.DUPLICATE_CLAIM)
        if not set(claims).issubset(required_ids):
            violations.add(TreatmentComplianceViolation.FOREIGN_CLAIM)
        if (
            request.requirement.claim_mode is TreatmentClaimMode.EXACT_REQUIRED
            and set(claims) != set(required_ids)
        ):
            violations.add(TreatmentComplianceViolation.EXACT_CLAIM_MISMATCH)
        if not request.operator_compliant:
            violations.add(TreatmentComplianceViolation.OPERATOR_NONCOMPLIANT)
        if request.selected_action not in request.preflight.compatible_actions:
            violations.add(TreatmentComplianceViolation.SELECTED_ACTION_INCOMPATIBLE)
        return TreatmentAdmissionReceipt(
            preflight_receipt_sha256=request.preflight.receipt_sha256,
            claimed_insight_ids=claims,
            selected_action=request.selected_action,
            violations=tuple(sorted(violations, key=str)),
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
        )


def validate_treatment_preflight_receipt(
    request: TreatmentPreflightRequest,
    receipt: TreatmentPreflightReceipt,
) -> None:
    """Independently enforce exact-action invariants on an injected policy."""

    if type(request) is not TreatmentPreflightRequest:
        raise TypeError("request must be an exact TreatmentPreflightRequest")
    TreatmentPreflightRequest.__post_init__(request)
    if type(receipt) is not TreatmentPreflightReceipt:
        raise TypeError("receipt must be an exact TreatmentPreflightReceipt")
    TreatmentPreflightReceipt.__post_init__(receipt)
    if receipt.requirement_sha256 != request.requirement.requirement_sha256:
        raise ValueError("preflight receipt changed the treatment requirement")
    if receipt.finite_contract_sha256 != request.finite_contract_sha256:
        raise ValueError("preflight receipt changed the finite contract")
    trusted_actions = set(request.actions)
    if any(action not in trusted_actions for action in receipt.compatible_actions):
        raise ValueError("preflight receipt introduced an untrusted finite action")
    if receipt.compatible_families != tuple(
        sorted({action.family for action in receipt.compatible_actions})
    ):
        raise ValueError("preflight receipt family projection is inconsistent")

    core_receipt = StrictTreatmentCompliancePolicy().preflight(request)
    core_violations = set(core_receipt.violations)
    if not core_violations.issubset(receipt.violations):
        raise ValueError("preflight receipt omitted a mandatory core violation")
    if receipt.passed and (
        not receipt.compatible_actions
        or not set(receipt.compatible_actions).issubset(
            core_receipt.compatible_actions
        )
    ):
        raise ValueError(
            "passing preflight must retain a nonempty trusted compatible subset"
        )


def validate_treatment_admission_receipt(
    request: TreatmentAdmissionRequest,
    receipt: TreatmentAdmissionReceipt,
) -> None:
    """Verify echoed proposal facts and mandatory pre-evaluator violations."""

    if type(request) is not TreatmentAdmissionRequest:
        raise TypeError("request must be an exact TreatmentAdmissionRequest")
    TreatmentAdmissionRequest.__post_init__(request)
    if type(receipt) is not TreatmentAdmissionReceipt:
        raise TypeError("receipt must be an exact TreatmentAdmissionReceipt")
    TreatmentAdmissionReceipt.__post_init__(receipt)
    if receipt.preflight_receipt_sha256 != request.preflight.receipt_sha256:
        raise ValueError("admission receipt changed the preflight identity")
    if receipt.claimed_insight_ids != request.claimed_insight_ids:
        raise ValueError("admission receipt changed the model's insight claims")
    if receipt.selected_action != request.selected_action:
        raise ValueError("admission receipt changed the selected finite action")

    core_receipt = StrictTreatmentCompliancePolicy().assess(request)
    if not set(core_receipt.violations).issubset(receipt.violations):
        raise ValueError("admission receipt omitted a mandatory core violation")


__all__ = [
    "FiniteTreatmentAction",
    "InsightTreatmentRequirement",
    "StrictTreatmentCompliancePolicy",
    "TREATMENT_COMPLIANCE_DEFINITION_SHA256",
    "TREATMENT_COMPLIANCE_POLICY_ID",
    "TREATMENT_COMPLIANCE_POLICY_VERSION",
    "TreatmentAdmissionReceipt",
    "TreatmentAdmissionRequest",
    "TreatmentActionBinding",
    "TreatmentAssignmentRole",
    "TreatmentClaimMode",
    "TreatmentCompliancePolicy",
    "TreatmentComplianceRejected",
    "TreatmentComplianceViolation",
    "TreatmentInsightEvidence",
    "TreatmentInsightBinding",
    "TreatmentPreflightReceipt",
    "TreatmentPreflightRequest",
    "validate_treatment_admission_receipt",
    "validate_treatment_preflight_receipt",
]
