"""Outcome-blind identification audit for anchor–residual portfolios.

Campaign runtimes already authenticate every selector request and structured
decision.  This module projects the regret-bounded allocation evidence from
those generic receipts, then assesses a workload- and model-independent
identification contract.  It never reads candidate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

from agent_evolve.application.campaign_execution import (
    CampaignSelectorAuditReceipt,
    CampaignStageReceipt,
    decode_selector_audit_text,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import thaw_json


REGRET_BOUNDED_SELECTOR_AUDIT_KIND = (
    "regret_bounded_information_portfolio_k8_to_k4"
)
STRICTLY_PRIOR_CERTIFICATE_SCOPE = (
    "conditional_on_frozen_acquisition_calibration_not_sota"
)


def _exact_option_ids(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise TypeError(f"{name} must be a non-empty option-id array")
    result = tuple(value)
    if any(type(option_id) is not str or not option_id for option_id in result):
        raise TypeError(f"{name} contains an invalid option identity")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate option identities")
    return result


def _hex_float(value: object, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a hexadecimal float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class AnchorResidualSelectionAudit:
    """One authenticated pre-evaluation reference displacement decision."""

    generation: int
    parent_slot: int
    selector_call_id: str
    request_sha256: str
    decision_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    certificate_scope: str
    reference_option_ids: tuple[str, ...]
    selected_option_ids: tuple[str, ...]
    reported_retained_anchor_member_count: int
    minimum_residual_audit_members: int
    reported_selected_residual_member_count: int
    acquisition_retention_ratio: float
    minimum_acquisition_retention_ratio: float

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be positive")
        if type(self.parent_slot) is not int or self.parent_slot < 0:
            raise ValueError("parent_slot must be non-negative")
        if type(self.selector_call_id) is not str or not self.selector_call_id:
            raise ValueError("selector_call_id must be non-empty")
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.decision_sha256, "decision_sha256")
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.certificate_scope) is not str or not self.certificate_scope:
            raise ValueError("certificate_scope must be non-empty")
        for name in ("reference_option_ids", "selected_option_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise TypeError(f"{name} must be a non-empty tuple")
            if any(type(value) is not str or not value for value in values):
                raise TypeError(f"{name} contains an invalid option identity")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate option identities")
        if len(self.reference_option_ids) != len(self.selected_option_ids):
            raise ValueError("reference and selected slate widths differ")
        for name in (
            "reported_retained_anchor_member_count",
            "minimum_residual_audit_members",
            "reported_selected_residual_member_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        retained = len(self.retained_anchor_option_ids)
        residual = len(self.residual_option_ids)
        if self.reported_retained_anchor_member_count != retained:
            raise ValueError("reported retained-anchor count does not close")
        if self.reported_selected_residual_member_count != residual:
            raise ValueError("reported residual count does not close")
        if retained + residual != len(self.selected_option_ids):
            raise ValueError("anchor–residual decomposition does not close")
        if len(self.displaced_anchor_option_ids) != residual:
            raise ValueError("residual and displaced-anchor counts differ")
        if self.minimum_residual_audit_members > residual:
            raise ValueError("selected slate violates its residual audit floor")
        for name in (
            "acquisition_retention_ratio",
            "minimum_acquisition_retention_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.minimum_acquisition_retention_ratio <= 1.0:
            raise ValueError("minimum acquisition retention must lie in (0, 1]")
        if not 0.0 < self.acquisition_retention_ratio <= 1.0:
            raise ValueError("acquisition retention must lie in (0, 1]")
        if not self.hard_retention_bound_respected:
            raise ValueError("selection escapes its hard retention envelope")

    @property
    def retained_anchor_option_ids(self) -> tuple[str, ...]:
        reference = set(self.reference_option_ids)
        return tuple(value for value in self.selected_option_ids if value in reference)

    @property
    def residual_option_ids(self) -> tuple[str, ...]:
        reference = set(self.reference_option_ids)
        return tuple(value for value in self.selected_option_ids if value not in reference)

    @property
    def displaced_anchor_option_ids(self) -> tuple[str, ...]:
        selected = set(self.selected_option_ids)
        return tuple(value for value in self.reference_option_ids if value not in selected)

    @property
    def hard_retention_bound_respected(self) -> bool:
        return (
            self.acquisition_retention_ratio
            >= self.minimum_acquisition_retention_ratio
        )

    @classmethod
    def from_allocation_record(
        cls,
        *,
        generation: int,
        parent_slot: int,
        selector_call_id: str,
        request_sha256: str,
        decision_sha256: str,
        allocation: dict[str, object],
    ) -> AnchorResidualSelectionAudit:
        if type(allocation) is not dict:
            raise TypeError("allocation must be an exact object")
        result = cls(
            generation=generation,
            parent_slot=parent_slot,
            selector_call_id=selector_call_id,
            request_sha256=request_sha256,
            decision_sha256=decision_sha256,
            policy_id=str(allocation.get("policy_id", "")),
            policy_version=allocation.get("policy_version"),
            policy_definition_sha256=allocation.get("policy_definition_sha256"),
            certificate_scope=str(allocation.get("certificate_scope", "")),
            reference_option_ids=_exact_option_ids(
                allocation.get("reference_option_ids"), "reference_option_ids"
            ),
            selected_option_ids=_exact_option_ids(
                allocation.get("selected_option_ids"), "selected_option_ids"
            ),
            reported_retained_anchor_member_count=allocation.get(
                "reference_member_count"
            ),
            minimum_residual_audit_members=allocation.get(
                "minimum_residual_audit_members"
            ),
            reported_selected_residual_member_count=allocation.get(
                "selected_residual_member_count"
            ),
            acquisition_retention_ratio=_hex_float(
                allocation.get("acquisition_retention_ratio_hex"),
                "acquisition_retention_ratio_hex",
            ),
            minimum_acquisition_retention_ratio=_hex_float(
                allocation.get("minimum_acquisition_retention_ratio_hex"),
                "minimum_acquisition_retention_ratio_hex",
            ),
        )
        result.__post_init__()
        return result

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "generation": self.generation,
            "parent_slot": self.parent_slot,
            "selector_call_id": self.selector_call_id,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "certificate_scope": self.certificate_scope,
            "reference_option_ids": list(self.reference_option_ids),
            "selected_option_ids": list(self.selected_option_ids),
            "retained_anchor_option_ids": list(self.retained_anchor_option_ids),
            "displaced_anchor_option_ids": list(self.displaced_anchor_option_ids),
            "residual_option_ids": list(self.residual_option_ids),
            "reference_slate_member_count": len(self.reference_option_ids),
            "retained_anchor_member_count": len(self.retained_anchor_option_ids),
            "displaced_anchor_member_count": len(
                self.displaced_anchor_option_ids
            ),
            "residual_member_count": len(self.residual_option_ids),
            "minimum_residual_audit_members": self.minimum_residual_audit_members,
            "acquisition_retention_ratio_hex": (
                self.acquisition_retention_ratio.hex()
            ),
            "minimum_acquisition_retention_ratio_hex": (
                self.minimum_acquisition_retention_ratio.hex()
            ),
            "hard_retention_bound_respected": (
                self.hard_retention_bound_respected
            ),
        }


def _supplemental_audit(
    receipt: CampaignSelectorAuditReceipt,
) -> dict[str, object] | None:
    receipt.__post_init__()
    plaintext = thaw_json(receipt.plaintext_audit)
    if type(plaintext) is not dict:  # pragma: no cover - closed by receipt.
        raise AssertionError("selector audit did not thaw to an object")
    response_text = decode_selector_audit_text(plaintext, "response_text")
    response = json.loads(response_text)
    if type(response) is not dict:
        raise TypeError("selector response audit must be an exact object")
    supplemental = response.get("supplemental_selector_audit")
    if supplemental is None:
        return None
    if type(supplemental) is not dict:
        raise TypeError("supplemental selector audit must be an exact object")
    if (
        supplemental.get("request_sha256") != receipt.request_sha256
        or supplemental.get("decision_sha256") != receipt.decision_sha256
    ):
        raise ValueError("supplemental audit differs from selector receipt identity")
    return supplemental


def project_anchor_residual_selection_audits(
    stage_receipts: tuple[CampaignStageReceipt, ...],
) -> tuple[AnchorResidualSelectionAudit, ...]:
    """Project all regret-bounded decisions from generic campaign receipts."""

    if type(stage_receipts) is not tuple or any(
        type(value) is not CampaignStageReceipt for value in stage_receipts
    ):
        raise TypeError("stage_receipts must contain exact stage receipts")
    records: list[AnchorResidualSelectionAudit] = []
    for stage in stage_receipts:
        stage.__post_init__()
        for receipt in stage.selector_audits:
            supplemental = _supplemental_audit(receipt)
            if supplemental is None:
                continue
            if supplemental.get("audit_kind") != REGRET_BOUNDED_SELECTOR_AUDIT_KIND:
                continue
            payload = supplemental.get("payload")
            allocation = payload.get("allocation") if type(payload) is dict else None
            if type(allocation) is not dict:
                raise TypeError("regret-bounded audit lacks allocation evidence")
            records.append(
                AnchorResidualSelectionAudit.from_allocation_record(
                    generation=receipt.generation,
                    parent_slot=receipt.parent_slot,
                    selector_call_id=receipt.selector_call_id,
                    request_sha256=receipt.request_sha256,
                    decision_sha256=receipt.decision_sha256,
                    allocation=allocation,
                )
            )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class AnchorResidualIdentificationContract:
    """Expected causal treatment shared by every workload and model."""

    expected_selector_calls: int
    portfolio_width: int
    minimum_residual_members: int
    exact_residual_members: int | None = None

    def __post_init__(self) -> None:
        if type(self.expected_selector_calls) is not int or (
            self.expected_selector_calls <= 0
        ):
            raise ValueError("expected_selector_calls must be positive")
        if type(self.portfolio_width) is not int or self.portfolio_width <= 1:
            raise ValueError("portfolio_width must exceed one")
        if (
            type(self.minimum_residual_members) is not int
            or not 0 <= self.minimum_residual_members < self.portfolio_width
        ):
            raise ValueError("minimum_residual_members lies outside the portfolio")
        if self.exact_residual_members is not None and (
            type(self.exact_residual_members) is not int
            or not self.minimum_residual_members
            <= self.exact_residual_members
            < self.portfolio_width
        ):
            raise ValueError("exact_residual_members lies outside the contract")

    def assess(
        self,
        records: tuple[AnchorResidualSelectionAudit, ...],
    ) -> AnchorResidualIdentificationAssessment:
        self.__post_init__()
        if type(records) is not tuple or any(
            type(value) is not AnchorResidualSelectionAudit for value in records
        ):
            raise TypeError("records must contain exact anchor-residual audits")
        for value in records:
            value.__post_init__()
        gates = {
            "exact_selector_call_count": (
                len(records) == self.expected_selector_calls
            ),
            "unique_authenticated_selector_calls": (
                len({value.selector_call_id for value in records}) == len(records)
                and len({value.request_sha256 for value in records}) == len(records)
                and len({value.decision_sha256 for value in records}) == len(records)
            ),
            "complete_reference_and_selected_slates": all(
                len(value.reference_option_ids) == self.portfolio_width
                and len(value.selected_option_ids) == self.portfolio_width
                for value in records
            ),
            "configured_residual_floor_realized": all(
                value.minimum_residual_audit_members
                == self.minimum_residual_members
                and len(value.residual_option_ids) >= self.minimum_residual_members
                for value in records
            ),
            "exact_residual_mix_realized": (
                self.exact_residual_members is None
                or all(
                    len(value.residual_option_ids) == self.exact_residual_members
                    and len(value.retained_anchor_option_ids)
                    == self.portfolio_width - self.exact_residual_members
                    for value in records
                )
            ),
            "hard_retention_bound_respected": all(
                value.hard_retention_bound_respected for value in records
            ),
            "strictly_prior_certificate_scope": all(
                value.certificate_scope == STRICTLY_PRIOR_CERTIFICATE_SCOPE
                for value in records
            ),
            "single_allocator_policy_identity": len(
                {
                    (
                        value.policy_id,
                        value.policy_version,
                        value.policy_definition_sha256,
                    )
                    for value in records
                }
            )
            == 1,
        }
        return AnchorResidualIdentificationAssessment(
            contract=self,
            records=records,
            gates=gates,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "expected_selector_calls": self.expected_selector_calls,
            "portfolio_width": self.portfolio_width,
            "minimum_residual_members": self.minimum_residual_members,
            "exact_residual_members": self.exact_residual_members,
        }


@dataclass(frozen=True, slots=True)
class AnchorResidualIdentificationAssessment:
    contract: AnchorResidualIdentificationContract
    records: tuple[AnchorResidualSelectionAudit, ...]
    gates: dict[str, bool]

    def __post_init__(self) -> None:
        if type(self.contract) is not AnchorResidualIdentificationContract:
            raise TypeError("contract must be exact")
        self.contract.__post_init__()
        if type(self.records) is not tuple or any(
            type(value) is not AnchorResidualSelectionAudit for value in self.records
        ):
            raise TypeError("records must contain exact anchor-residual audits")
        if type(self.gates) is not dict or not self.gates:
            raise TypeError("gates must be a non-empty exact object")
        if any(type(name) is not str or type(value) is not bool for name, value in self.gates.items()):
            raise TypeError("gates must map strings to exact bools")

    @property
    def all_gates_pass(self) -> bool:
        return all(self.gates.values())

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        ratios = tuple(value.acquisition_retention_ratio for value in self.records)
        return {
            "contract": self.contract.to_record(),
            "all_gates_pass": self.all_gates_pass,
            "gates": dict(self.gates),
            "selector_call_count": len(self.records),
            "retained_anchor_member_count": sum(
                len(value.retained_anchor_option_ids) for value in self.records
            ),
            "residual_member_count": sum(
                len(value.residual_option_ids) for value in self.records
            ),
            "minimum_realized_acquisition_retention_ratio_hex": (
                None if not ratios else min(ratios).hex()
            ),
            "maximum_realized_acquisition_retention_ratio_hex": (
                None if not ratios else max(ratios).hex()
            ),
            "records": [value.to_record() for value in self.records],
        }


__all__ = [
    "AnchorResidualIdentificationAssessment",
    "AnchorResidualIdentificationContract",
    "AnchorResidualSelectionAudit",
    "REGRET_BOUNDED_SELECTOR_AUDIT_KIND",
    "STRICTLY_PRIOR_CERTIFICATE_SCOPE",
    "project_anchor_residual_selection_audits",
]
