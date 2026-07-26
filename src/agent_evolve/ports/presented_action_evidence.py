"""Provider-neutral contracts for prompt-visible evidence consistency.

These values describe evidence that a benchmark has already authenticated and
presented to an action-forecast request.  They do not authorize this package to
open an outcome store, reinterpret a benchmark receipt, or treat a presented
value as ground truth.  The application assessor only joins these immutable
values to an exact resolved forecast frame.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")

_CELL_DOMAIN = b"agent-evolve:presented-action-evidence-cell:v1\x00"
_POLICY_DOMAIN = b"agent-evolve:presented-action-evidence-consistency-policy:v1\x00"
_SUBSET_POLICY_DOMAIN = (
    b"agent-evolve:presented-action-evidence-subset-policy:v1\x00"
)
_SUBSET_DOMAIN = b"agent-evolve:presented-action-evidence-subset:v1\x00"
_CELL_ASSESSMENT_DOMAIN = (
    b"agent-evolve:presented-action-evidence-cell-assessment:v1\x00"
)
_CELL_SET_DOMAIN = b"agent-evolve:presented-action-evidence-cell-set:v1\x00"
_ASSESSMENT_DOMAIN = (
    b"agent-evolve:presented-action-evidence-consistency-assessment:v1\x00"
)

PRESENTED_ACTION_EVIDENCE_CONSISTENCY_SCOPE = (
    "presented_prompt_evidence_consistency_not_outcome_truth_or_calibration"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    return value


def _token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")
    return value


def _option_id(value: object, name: str) -> str:
    if type(value) is not str or _OPTION_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed option grammar")
    return value


def _metric_id(value: object, name: str) -> str:
    if type(value) is not str or _METRIC_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed metric grammar")
    return value


class PresentedActionEvidenceProvenanceKind(str, Enum):
    """Closed prompt-view provenance that can authenticate one cell."""

    CARD_SOURCE_RECEIPT = "card_source_receipt"
    CARD_VIEW_RECEIPT = "card_view_receipt"
    REQUEST_CARD_SNAPSHOT = "request_card_snapshot"


class PresentedActionEvidenceConsistencyFrameKind(str, Enum):
    BLOCK = "block"
    SUBSET = "subset"


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceCell:
    """One benchmark-injected value exactly as represented to the model.

    ``option_identity_sha256`` identifies the current forecast row to which
    the benchmark maps the presented observation.  It need not equal the
    historical option identity inside the action-evidence binding.  Keeping
    both identities explicit prevents a benchmark mapping from being inferred
    from names, families, or domain-specific conventions.
    """

    option_identity_sha256: str
    metric_id: str
    presented_delta: float
    card_key: str
    action_evidence_binding_identity_sha256: str
    provenance_kind: PresentedActionEvidenceProvenanceKind
    provenance_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        _metric_id(self.metric_id, "metric_id")
        _finite_float(self.presented_delta, "presented_delta")
        _token(self.card_key, "card_key")
        require_sha256(
            self.action_evidence_binding_identity_sha256,
            "action_evidence_binding_identity_sha256",
        )
        if type(self.provenance_kind) is not PresentedActionEvidenceProvenanceKind:
            raise TypeError(
                "provenance_kind must be an exact presented-evidence provenance kind"
            )
        require_sha256(self.provenance_sha256, "provenance_sha256")

    @property
    def join_key(self) -> tuple[str, ...]:
        self.__post_init__()
        return (
            self.option_identity_sha256,
            self.metric_id,
            self.card_key,
            self.action_evidence_binding_identity_sha256,
        )

    @property
    def sort_key(self) -> tuple[str, ...]:
        self.__post_init__()
        return (
            *self.join_key,
            self.provenance_kind.value,
            self.provenance_sha256,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "option_identity_sha256": self.option_identity_sha256,
            "metric_id": self.metric_id,
            "presented_delta_hex": self.presented_delta.hex(),
            "card_key": self.card_key,
            "action_evidence_binding_identity_sha256": (
                self.action_evidence_binding_identity_sha256
            ),
            "provenance_kind": self.provenance_kind.value,
            "provenance_sha256": self.provenance_sha256,
        }

    @property
    def cell_sha256(self) -> str:
        return _hash(_CELL_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "cell_sha256": self.cell_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceCell
            and type(other) is PresentedActionEvidenceCell
            and self.cell_sha256 == other.cell_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceConsistencyPolicyBinding:
    """Identified descriptive or fail-closed consistency decision policy.

    Provenance receipts are mandatory for every policy.  With all decision
    controls disabled, the assessment is descriptive and ``passes`` is
    ``None``.  A later experiment can inject an identified policy that enables
    any combination of the three fail-closed, all-cell requirements.
    """

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    maximum_normalized_absolute_error: float | None
    require_direction_agreement: bool
    require_interval_coverage: bool

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if self.maximum_normalized_absolute_error is not None:
            _finite_float(
                self.maximum_normalized_absolute_error,
                "maximum_normalized_absolute_error",
            )
            if self.maximum_normalized_absolute_error < 0.0:
                raise ValueError(
                    "maximum_normalized_absolute_error cannot be negative"
                )
        if type(self.require_direction_agreement) is not bool:
            raise TypeError("require_direction_agreement must be an exact boolean")
        if type(self.require_interval_coverage) is not bool:
            raise TypeError("require_interval_coverage must be an exact boolean")

    @property
    def decision_applied(self) -> bool:
        self.__post_init__()
        return (
            self.maximum_normalized_absolute_error is not None
            or self.require_direction_agreement
            or self.require_interval_coverage
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "authenticated_provenance_required": True,
            "maximum_normalized_absolute_error_hex": (
                None
                if self.maximum_normalized_absolute_error is None
                else self.maximum_normalized_absolute_error.hex()
            ),
            "require_direction_agreement": self.require_direction_agreement,
            "require_interval_coverage": self.require_interval_coverage,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceConsistencyPolicyBinding
            and type(other) is PresentedActionEvidenceConsistencyPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceSubsetPolicyBinding:
    """Identified benchmark-neutral rule selecting rows from one block."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        _token(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_SUBSET_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceSubsetPolicyBinding
            and type(other) is PresentedActionEvidenceSubsetPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceSubsetBinding:
    """Authenticated ordered row subset of one exact forecast block receipt."""

    subset_policy: PresentedActionEvidenceSubsetPolicyBinding
    request_sha256: str
    layout_sha256: str
    block_request_sha256: str
    block_spec_sha256: str
    forecast_block_receipt_sha256: str
    block_index: int
    included_global_row_indices: tuple[int, ...]
    included_option_identity_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.subset_policy) is not PresentedActionEvidenceSubsetPolicyBinding:
            raise TypeError("subset_policy must be an exact subset-policy binding")
        self.subset_policy.__post_init__()
        for name in (
            "request_sha256",
            "layout_sha256",
            "block_request_sha256",
            "block_spec_sha256",
            "forecast_block_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        if type(self.included_global_row_indices) is not tuple or not (
            self.included_global_row_indices
        ) or any(
            type(value) is not int for value in self.included_global_row_indices
        ):
            raise ValueError(
                "included_global_row_indices must be a non-empty exact tuple"
            )
        if self.included_global_row_indices != tuple(
            sorted(set(self.included_global_row_indices))
        ):
            raise ValueError("included global rows must be unique and canonical")
        if any(value < 0 for value in self.included_global_row_indices):
            raise ValueError("included global rows cannot be negative")
        if type(self.included_option_identity_sha256s) is not tuple or len(
            self.included_option_identity_sha256s
        ) != len(self.included_global_row_indices):
            raise ValueError(
                "included option identities must exactly cover included rows"
            )
        for index, value in enumerate(self.included_option_identity_sha256s):
            require_sha256(value, f"included_option_identity_sha256s[{index}]")
        if len(set(self.included_option_identity_sha256s)) != len(
            self.included_option_identity_sha256s
        ):
            raise ValueError("a subset cannot repeat an option identity")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "subset_policy": self.subset_policy.to_record(),
            "request_sha256": self.request_sha256,
            "layout_sha256": self.layout_sha256,
            "block_request_sha256": self.block_request_sha256,
            "block_spec_sha256": self.block_spec_sha256,
            "forecast_block_receipt_sha256": (
                self.forecast_block_receipt_sha256
            ),
            "block_index": self.block_index,
            "included_global_row_indices": list(
                self.included_global_row_indices
            ),
            "included_option_identity_sha256s": list(
                self.included_option_identity_sha256s
            ),
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_SUBSET_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceSubsetBinding
            and type(other) is PresentedActionEvidenceSubsetBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceCellAssessment:
    """One exact presented-value/forecast join with complete provenance."""

    presented_cell_sha256: str
    global_row_index: int
    option_id: str
    option_identity_sha256: str
    metric_id: str
    presented_delta: float
    p10_delta: float
    p50_delta: float
    p90_delta: float
    direction_agreement: bool
    interval_coverage: bool
    normalized_absolute_error: float
    metric_delta_scale: float
    metric_scale_definition_sha256: str
    card_key: str
    card_source_binding_sha256: str
    card_source_receipt_sha256: str
    card_view_receipt_sha256: str | None
    action_evidence_binding_identity_sha256: str
    source_contrast_id: str
    source_option_id: str
    source_family: str
    source_option_identity_sha256: str
    source_contract_identity_sha256: str
    provenance_kind: PresentedActionEvidenceProvenanceKind
    provenance_sha256: str
    forecast_cites_presented_binding: bool

    def __post_init__(self) -> None:
        for name in (
            "presented_cell_sha256",
            "option_identity_sha256",
            "metric_scale_definition_sha256",
            "card_source_binding_sha256",
            "card_source_receipt_sha256",
            "action_evidence_binding_identity_sha256",
            "source_contrast_id",
            "source_option_identity_sha256",
            "source_contract_identity_sha256",
            "provenance_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.card_view_receipt_sha256 is not None:
            require_sha256(
                self.card_view_receipt_sha256,
                "card_view_receipt_sha256",
            )
        if type(self.global_row_index) is not int or self.global_row_index < 0:
            raise ValueError("global_row_index must be a non-negative exact integer")
        _option_id(self.option_id, "option_id")
        _metric_id(self.metric_id, "metric_id")
        _token(self.card_key, "card_key")
        _option_id(self.source_option_id, "source_option_id")
        _token(self.source_family, "source_family")
        for name in (
            "presented_delta",
            "p10_delta",
            "p50_delta",
            "p90_delta",
            "normalized_absolute_error",
            "metric_delta_scale",
        ):
            _finite_float(getattr(self, name), name)
        if not self.p10_delta <= self.p50_delta <= self.p90_delta:
            raise ValueError("forecast quantiles must satisfy p10 <= p50 <= p90")
        if self.normalized_absolute_error < 0.0:
            raise ValueError("normalized_absolute_error cannot be negative")
        if self.metric_delta_scale <= 0.0:
            raise ValueError("metric_delta_scale must be positive")
        for name in (
            "direction_agreement",
            "interval_coverage",
            "forecast_cites_presented_binding",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if type(self.provenance_kind) is not PresentedActionEvidenceProvenanceKind:
            raise TypeError(
                "provenance_kind must be an exact presented-evidence provenance kind"
            )
        expected_cell_sha256 = PresentedActionEvidenceCell(
            option_identity_sha256=self.option_identity_sha256,
            metric_id=self.metric_id,
            presented_delta=self.presented_delta,
            card_key=self.card_key,
            action_evidence_binding_identity_sha256=(
                self.action_evidence_binding_identity_sha256
            ),
            provenance_kind=self.provenance_kind,
            provenance_sha256=self.provenance_sha256,
        ).cell_sha256
        if self.presented_cell_sha256 != expected_cell_sha256:
            raise ValueError(
                "presented_cell_sha256 differs from the assessed cell fields"
            )
        expected_direction = (
            self.p50_delta == self.presented_delta
            if self.p50_delta == 0.0 or self.presented_delta == 0.0
            else (self.p50_delta > 0.0) == (self.presented_delta > 0.0)
        )
        if self.direction_agreement != expected_direction:
            raise ValueError("direction_agreement differs from the exact deltas")
        expected_coverage = (
            self.p10_delta <= self.presented_delta <= self.p90_delta
        )
        if self.interval_coverage != expected_coverage:
            raise ValueError("interval_coverage differs from the exact quantiles")
        expected_error = (
            abs(self.p50_delta - self.presented_delta) / self.metric_delta_scale
        )
        if self.normalized_absolute_error != expected_error:
            raise ValueError(
                "normalized_absolute_error differs from the exact scale join"
            )

    @property
    def join_key(self) -> tuple[object, ...]:
        self.__post_init__()
        return (
            self.global_row_index,
            self.metric_id,
            self.card_key,
            self.action_evidence_binding_identity_sha256,
        )

    @property
    def sort_key(self) -> tuple[object, ...]:
        self.__post_init__()
        return (
            *self.join_key,
            self.provenance_kind.value,
            self.provenance_sha256,
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "presented_cell_sha256": self.presented_cell_sha256,
            "global_row_index": self.global_row_index,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "metric_id": self.metric_id,
            "presented_delta_hex": self.presented_delta.hex(),
            "forecast_quantiles": {
                "p10_delta_hex": self.p10_delta.hex(),
                "p50_delta_hex": self.p50_delta.hex(),
                "p90_delta_hex": self.p90_delta.hex(),
            },
            "direction_agreement": self.direction_agreement,
            "interval_coverage": self.interval_coverage,
            "normalized_absolute_error_hex": (
                self.normalized_absolute_error.hex()
            ),
            "metric_scale": {
                "delta_scale_hex": self.metric_delta_scale.hex(),
                "definition_sha256": self.metric_scale_definition_sha256,
            },
            "provenance": {
                "card_key": self.card_key,
                "card_source_binding_sha256": (
                    self.card_source_binding_sha256
                ),
                "card_source_receipt_sha256": (
                    self.card_source_receipt_sha256
                ),
                "card_view_receipt_sha256": self.card_view_receipt_sha256,
                "action_evidence_binding_identity_sha256": (
                    self.action_evidence_binding_identity_sha256
                ),
                "source_contrast_id": self.source_contrast_id,
                "source_option_id": self.source_option_id,
                "source_family": self.source_family,
                "source_option_identity_sha256": (
                    self.source_option_identity_sha256
                ),
                "source_contract_identity_sha256": (
                    self.source_contract_identity_sha256
                ),
                "provenance_kind": self.provenance_kind.value,
                "provenance_sha256": self.provenance_sha256,
            },
            "forecast_cites_presented_binding": (
                self.forecast_cites_presented_binding
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CELL_ASSESSMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceCellAssessment
            and type(other) is PresentedActionEvidenceCellAssessment
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PresentedActionEvidenceConsistencyAssessment:
    """Authenticated descriptive assessment for one exact block or subset."""

    frame_kind: PresentedActionEvidenceConsistencyFrameKind
    request_sha256: str
    request_card_snapshot_sha256: str
    experimental_view_receipt_sha256: str
    layout_sha256: str
    block_request_sha256: str
    block_spec_sha256: str
    block_index: int
    forecast_block_receipt_sha256: str
    subset_binding: PresentedActionEvidenceSubsetBinding | None
    policy: PresentedActionEvidenceConsistencyPolicyBinding
    cell_assessments: tuple[PresentedActionEvidenceCellAssessment, ...]
    decision_applied: bool
    passes: bool | None

    def __post_init__(self) -> None:
        if type(self.frame_kind) is not PresentedActionEvidenceConsistencyFrameKind:
            raise TypeError("frame_kind must be an exact consistency-frame kind")
        for name in (
            "request_sha256",
            "request_card_snapshot_sha256",
            "experimental_view_receipt_sha256",
            "layout_sha256",
            "block_request_sha256",
            "block_spec_sha256",
            "forecast_block_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        if self.frame_kind is PresentedActionEvidenceConsistencyFrameKind.BLOCK:
            if self.subset_binding is not None:
                raise ValueError("block assessments forbid subset bindings")
        else:
            if type(self.subset_binding) is not PresentedActionEvidenceSubsetBinding:
                raise TypeError("subset assessments require an exact subset binding")
            self.subset_binding.__post_init__()
            if (
                self.subset_binding.request_sha256 != self.request_sha256
                or self.subset_binding.layout_sha256 != self.layout_sha256
                or self.subset_binding.block_request_sha256
                != self.block_request_sha256
                or self.subset_binding.block_spec_sha256 != self.block_spec_sha256
                or self.subset_binding.block_index != self.block_index
                or self.subset_binding.forecast_block_receipt_sha256
                != self.forecast_block_receipt_sha256
            ):
                raise ValueError("subset binding differs from the assessment frame")
        if type(self.policy) is not PresentedActionEvidenceConsistencyPolicyBinding:
            raise TypeError("policy must be an exact consistency-policy binding")
        self.policy.__post_init__()
        if type(self.cell_assessments) is not tuple or not self.cell_assessments or any(
            type(value) is not PresentedActionEvidenceCellAssessment
            for value in self.cell_assessments
        ):
            raise ValueError("cell_assessments must be a non-empty exact tuple")
        for value in self.cell_assessments:
            value.__post_init__()
        keys = tuple(value.sort_key for value in self.cell_assessments)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("cell assessments must be unique and canonical")
        join_keys = tuple(value.join_key for value in self.cell_assessments)
        if len(set(join_keys)) != len(join_keys):
            raise ValueError("cell assessments cannot repeat an evidence join")
        if type(self.decision_applied) is not bool:
            raise TypeError("decision_applied must be an exact boolean")
        if self.decision_applied != self.policy.decision_applied:
            raise ValueError("decision_applied differs from the identified policy")
        if self.decision_applied:
            if type(self.passes) is not bool:
                raise TypeError("an applied policy requires an exact pass decision")
            expected = all(
                (
                    self.policy.maximum_normalized_absolute_error is None
                    or cell.normalized_absolute_error
                    <= self.policy.maximum_normalized_absolute_error
                )
                and (
                    not self.policy.require_direction_agreement
                    or cell.direction_agreement
                )
                and (
                    not self.policy.require_interval_coverage
                    or cell.interval_coverage
                )
                for cell in self.cell_assessments
            )
            if self.passes != expected:
                raise ValueError("pass decision differs from the identified policy")
        elif self.passes is not None:
            raise ValueError("a descriptive policy must not publish pass/fail")

    @property
    def cell_set_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            _CELL_SET_DOMAIN,
            {
                "schema_version": 1,
                "presented_cell_sha256s": [
                    value.presented_cell_sha256 for value in self.cell_assessments
                ],
            },
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        count = len(self.cell_assessments)
        return {
            "schema_version": 1,
            "scientific_scope": PRESENTED_ACTION_EVIDENCE_CONSISTENCY_SCOPE,
            "frame_kind": self.frame_kind.value,
            "request_sha256": self.request_sha256,
            "request_card_snapshot_sha256": (
                self.request_card_snapshot_sha256
            ),
            "experimental_view_receipt_sha256": (
                self.experimental_view_receipt_sha256
            ),
            "layout_sha256": self.layout_sha256,
            "block_request_sha256": self.block_request_sha256,
            "block_spec_sha256": self.block_spec_sha256,
            "block_index": self.block_index,
            "forecast_block_receipt_sha256": (
                self.forecast_block_receipt_sha256
            ),
            "subset_binding": (
                None
                if self.subset_binding is None
                else self.subset_binding.to_record()
            ),
            "policy": self.policy.to_record(),
            "cell_set_sha256": self.cell_set_sha256,
            "cell_assessments": [
                value.to_record() for value in self.cell_assessments
            ],
            "summary": {
                "cell_count": count,
                "mean_normalized_absolute_error_hex": (
                    sum(
                        value.normalized_absolute_error
                        for value in self.cell_assessments
                    )
                    / count
                ).hex(),
                "maximum_normalized_absolute_error_hex": max(
                    value.normalized_absolute_error
                    for value in self.cell_assessments
                ).hex(),
                "direction_agreement_share_hex": (
                    sum(value.direction_agreement for value in self.cell_assessments)
                    / count
                ).hex(),
                "interval_coverage_share_hex": (
                    sum(value.interval_coverage for value in self.cell_assessments)
                    / count
                ).hex(),
            },
            "decision_applied": self.decision_applied,
            "passes": self.passes,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_ASSESSMENT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PresentedActionEvidenceConsistencyAssessment
            and type(other) is PresentedActionEvidenceConsistencyAssessment
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


__all__ = [
    "PRESENTED_ACTION_EVIDENCE_CONSISTENCY_SCOPE",
    "PresentedActionEvidenceCell",
    "PresentedActionEvidenceCellAssessment",
    "PresentedActionEvidenceConsistencyAssessment",
    "PresentedActionEvidenceConsistencyFrameKind",
    "PresentedActionEvidenceConsistencyPolicyBinding",
    "PresentedActionEvidenceProvenanceKind",
    "PresentedActionEvidenceSubsetBinding",
    "PresentedActionEvidenceSubsetPolicyBinding",
]
