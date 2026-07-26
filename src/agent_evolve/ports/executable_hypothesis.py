"""Provider-neutral compilation of memory hypotheses into executable tests.

The memory bank owns immutable, evidence-lined hypotheses.  A benchmark owns
the meaning of an action template in its current parent-bound variation
catalog.  This port is the dependency-inversion seam between those two facts:
trusted benchmark code may bind a hypothesis to exact finite options, but it
cannot edit the hypothesis, relax the catalog, or observe candidate outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.agentic_generator import InsightDraft, MetricEffectPrediction
from agent_evolve.policies.memory.treatment_compliance import TreatmentActionBinding


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_COMPILATION_DOMAIN = b"agent-evolve:hypothesis-compilation:v1\x00"
_SPEC_DOMAIN = b"agent-evolve:executable-hypothesis-test-spec:v1\x00"


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


def _canonical_tokens(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _TOKEN.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must contain canonical token strings")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _canonical_paths(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _PATH.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must contain canonical rooted JSON paths")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _path_segments(value: str) -> tuple[str, ...]:
    """Return comparison segments for a canonical rooted JSON path.

    The executable-hypothesis boundary does not interpret candidate values, but
    it must reject a compiler claiming that ``$.model`` is held fixed while
    mutating ``$.model.width`` (and the converse).  Separators are retained only
    as structural boundaries; quoted/escaped JSONPath dialects are deliberately
    outside this closed contract.
    """

    if _PATH.fullmatch(value) is None:
        raise ValueError("path must be a canonical rooted JSON path")
    tail = value[1:]
    return tuple(
        part
        for part in re.split(r"(?:\.|\[|\])", tail)
        if part
    )


def _paths_overlap(left: str, right: str) -> bool:
    left_segments = _path_segments(left)
    right_segments = _path_segments(right)
    common = min(len(left_segments), len(right_segments))
    return left_segments[:common] == right_segments[:common]


@dataclass(frozen=True, slots=True)
class HypothesisCompilationRequest:
    """Outcome-blind request to bind one exact hypothesis to one parent."""

    reference: InsightRef
    insight: InsightDraft
    source_evidence_sha256: str
    requested_operator_kind: str
    source_operator_kinds: tuple[str, ...]
    parent_candidate_id: CandidateId
    parent_configuration_sha256: str
    finite_contract: FiniteVariationContract
    context_projection_sha256: str
    endpoint_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        if type(self.insight) is not InsightDraft:
            raise TypeError("insight must be an exact InsightDraft")
        InsightDraft.__post_init__(self.insight)
        if not self.insight.has_intervention_contract:
            raise ValueError(
                "hypothesis compilation requires an intervention-contract insight"
            )
        require_sha256(self.source_evidence_sha256, "source_evidence_sha256")
        if (
            type(self.requested_operator_kind) is not str
            or _TOKEN.fullmatch(self.requested_operator_kind) is None
        ):
            raise ValueError("requested_operator_kind must use the token grammar")
        _canonical_tokens(
            self.source_operator_kinds,
            name="source_operator_kinds",
        )
        if not self.source_operator_kinds:
            raise ValueError("source_operator_kinds must be non-empty")
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.parent_candidate_id)
        require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        validate_finite_variation_contract(self.finite_contract)
        if (
            self.finite_contract.parent_configuration_sha256
            != self.parent_configuration_sha256
        ):
            raise ValueError("finite contract is bound to a different parent")
        require_sha256(
            self.context_projection_sha256,
            "context_projection_sha256",
        )
        require_sha256(
            self.endpoint_definition_sha256,
            "endpoint_definition_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "insight_content_sha256": self.insight.content_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "requested_operator_kind": self.requested_operator_kind,
            "source_operator_kinds": list(self.source_operator_kinds),
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "finite_contract_sha256": self.finite_contract.identity_sha256,
            "context_projection_sha256": self.context_projection_sha256,
            "endpoint_definition_sha256": self.endpoint_definition_sha256,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_COMPILATION_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class ExecutableHypothesisTestSpec:
    """Exact parent-bound intervention and falsification contract."""

    request_sha256: str
    reference: InsightRef
    insight_content_sha256: str
    source_evidence_sha256: str
    requested_operator_kind: str
    source_operator_kinds: tuple[str, ...]
    executable_operator_kinds: tuple[str, ...]
    parent_candidate_id: CandidateId
    parent_configuration_sha256: str
    finite_contract_sha256: str
    context_projection_sha256: str
    endpoint_definition_sha256: str
    allowed_actions: tuple[TreatmentActionBinding, ...]
    recommended_option_families: tuple[str, ...]
    affected_paths: tuple[str, ...]
    held_fixed_paths: tuple[str, ...]
    effect_predictions: tuple[MetricEffectPrediction, ...]
    falsification_condition: str
    compiler_policy_id: str
    compiler_policy_version: int
    compiler_definition_sha256: str
    spec_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "insight_content_sha256",
            "source_evidence_sha256",
            "parent_configuration_sha256",
            "finite_contract_sha256",
            "context_projection_sha256",
            "endpoint_definition_sha256",
            "compiler_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        if (
            type(self.requested_operator_kind) is not str
            or _TOKEN.fullmatch(self.requested_operator_kind) is None
        ):
            raise ValueError("requested_operator_kind must use the token grammar")
        _canonical_tokens(
            self.source_operator_kinds,
            name="source_operator_kinds",
        )
        if not self.source_operator_kinds:
            raise ValueError("source_operator_kinds must be non-empty")
        _canonical_tokens(
            self.executable_operator_kinds,
            name="executable_operator_kinds",
        )
        if self.executable_operator_kinds != (self.requested_operator_kind,):
            raise ValueError(
                "executable operator scope must equal the requested singleton"
            )
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.parent_candidate_id)
        if type(self.allowed_actions) is not tuple or not self.allowed_actions:
            raise ValueError("an executable hypothesis needs allowed actions")
        if any(
            type(value) is not TreatmentActionBinding
            for value in self.allowed_actions
        ):
            raise TypeError("allowed_actions must contain exact action bindings")
        for action in self.allowed_actions:
            TreatmentActionBinding.__post_init__(action)
        if self.allowed_actions != tuple(
            sorted(
                set(self.allowed_actions),
                key=lambda value: (value.option_id, value.option_identity_sha256),
            )
        ):
            raise ValueError("allowed_actions must be unique and canonical")
        if len({value.option_id for value in self.allowed_actions}) != len(
            self.allowed_actions
        ):
            raise ValueError("allowed action option IDs must be unique")
        _canonical_tokens(
            self.recommended_option_families,
            name="recommended_option_families",
        )
        if not self.recommended_option_families:
            raise ValueError("an executable hypothesis needs an option family")
        _canonical_paths(self.affected_paths, name="affected_paths")
        _canonical_paths(self.held_fixed_paths, name="held_fixed_paths")
        if not self.affected_paths:
            raise ValueError("an executable hypothesis needs affected paths")
        if any(
            _paths_overlap(affected, held_fixed)
            for affected in self.affected_paths
            for held_fixed in self.held_fixed_paths
        ):
            raise ValueError(
                "affected and held-fixed paths must be hierarchically disjoint"
            )
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not MetricEffectPrediction
            for value in self.effect_predictions
        ):
            raise TypeError("effect_predictions must contain exact predictions")
        for prediction in self.effect_predictions:
            MetricEffectPrediction.__post_init__(prediction)
        metric_ids = tuple(value.metric_id for value in self.effect_predictions)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("effect predictions must be unique and canonical")
        if (
            type(self.falsification_condition) is not str
            or not self.falsification_condition.strip()
            or self.falsification_condition != self.falsification_condition.strip()
        ):
            raise ValueError("falsification_condition must be canonical text")
        if (
            type(self.compiler_policy_id) is not str
            or _TOKEN.fullmatch(self.compiler_policy_id) is None
        ):
            raise ValueError("compiler_policy_id must use the token grammar")
        if (
            type(self.compiler_policy_version) is not int
            or self.compiler_policy_version <= 0
        ):
            raise ValueError("compiler_policy_version must be positive")
        object.__setattr__(self, "spec_sha256", _hash(_SPEC_DOMAIN, self.to_record()))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "insight_content_sha256": self.insight_content_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "requested_operator_kind": self.requested_operator_kind,
            "source_operator_kinds": list(self.source_operator_kinds),
            "executable_operator_kinds": list(self.executable_operator_kinds),
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "finite_contract_sha256": self.finite_contract_sha256,
            "context_projection_sha256": self.context_projection_sha256,
            "endpoint_definition_sha256": self.endpoint_definition_sha256,
            "allowed_actions": [value.to_record() for value in self.allowed_actions],
            "recommended_option_families": list(self.recommended_option_families),
            "affected_paths": list(self.affected_paths),
            "held_fixed_paths": list(self.held_fixed_paths),
            "effect_predictions": [
                {
                    "metric_id": value.metric_id,
                    "direction": value.direction.value,
                }
                for value in self.effect_predictions
            ],
            "falsification_condition": self.falsification_condition,
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_version": self.compiler_policy_version,
            "compiler_definition_sha256": self.compiler_definition_sha256,
        }


class HypothesisApplicabilityStatus(str, Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"


@dataclass(frozen=True, slots=True)
class HypothesisCompilationReceipt:
    """Fail-closed compiler result; only applicable receipts carry a spec."""

    request_sha256: str
    status: HypothesisApplicabilityStatus
    reason_codes: tuple[str, ...]
    compiler_policy_id: str
    compiler_policy_version: int
    compiler_definition_sha256: str
    spec: ExecutableHypothesisTestSpec | None
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.status) is not HypothesisApplicabilityStatus:
            raise TypeError("status must be a HypothesisApplicabilityStatus")
        _canonical_tokens(self.reason_codes, name="reason_codes")
        if (
            type(self.compiler_policy_id) is not str
            or _TOKEN.fullmatch(self.compiler_policy_id) is None
        ):
            raise ValueError("compiler_policy_id must use the token grammar")
        if (
            type(self.compiler_policy_version) is not int
            or self.compiler_policy_version <= 0
        ):
            raise ValueError("compiler_policy_version must be positive")
        require_sha256(
            self.compiler_definition_sha256,
            "compiler_definition_sha256",
        )
        if self.status is HypothesisApplicabilityStatus.APPLICABLE:
            if type(self.spec) is not ExecutableHypothesisTestSpec:
                raise ValueError("an applicable receipt requires an exact spec")
            ExecutableHypothesisTestSpec.__post_init__(self.spec)
            if self.reason_codes:
                raise ValueError("an applicable receipt cannot carry failure reasons")
            if (
                self.spec.request_sha256 != self.request_sha256
                or self.spec.compiler_policy_id != self.compiler_policy_id
                or self.spec.compiler_policy_version != self.compiler_policy_version
                or self.spec.compiler_definition_sha256
                != self.compiler_definition_sha256
            ):
                raise ValueError("compiler receipt and executable spec differ")
        elif self.spec is not None or not self.reason_codes:
            raise ValueError("an inapplicable receipt needs reasons and no spec")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_COMPILATION_DOMAIN, self.to_record()),
        )

    @property
    def applicable(self) -> bool:
        return self.status is HypothesisApplicabilityStatus.APPLICABLE

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_version": self.compiler_policy_version,
            "compiler_definition_sha256": self.compiler_definition_sha256,
            "spec": None if self.spec is None else self.spec.to_record(),
            "spec_sha256": None if self.spec is None else self.spec.spec_sha256,
        }


@runtime_checkable
class HypothesisApplicabilityPort(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt: ...


def validate_hypothesis_compiler_identity(
    compiler: HypothesisApplicabilityPort,
    receipt: HypothesisCompilationReceipt,
) -> None:
    """Bind a receipt to the injected compiler instance at the trust boundary."""

    if not isinstance(compiler, HypothesisApplicabilityPort):
        raise TypeError("compiler must implement HypothesisApplicabilityPort")
    if type(receipt) is not HypothesisCompilationReceipt:
        raise TypeError("receipt must be an exact HypothesisCompilationReceipt")
    receipt.__post_init__()
    observed = (
        compiler.policy_id,
        compiler.policy_version,
        compiler.definition_sha256,
    )
    expected = (
        receipt.compiler_policy_id,
        receipt.compiler_policy_version,
        receipt.compiler_definition_sha256,
    )
    if observed != expected:
        raise ValueError("compiler identity changed across compilation")


def validate_hypothesis_compilation(
    request: HypothesisCompilationRequest,
    receipt: HypothesisCompilationReceipt,
) -> None:
    """Independently bind compiler output to the trusted request and catalog."""

    if type(request) is not HypothesisCompilationRequest:
        raise TypeError("request must be an exact HypothesisCompilationRequest")
    request.__post_init__()
    if type(receipt) is not HypothesisCompilationReceipt:
        raise TypeError("receipt must be an exact HypothesisCompilationReceipt")
    receipt.__post_init__()
    if receipt.request_sha256 != request.request_sha256:
        raise ValueError("compiler receipt is bound to a different request")
    if not receipt.applicable:
        return
    spec = receipt.spec
    assert spec is not None
    expected = (
        request.reference,
        request.insight.content_sha256,
        request.source_evidence_sha256,
        request.requested_operator_kind,
        request.source_operator_kinds,
        request.parent_candidate_id,
        request.parent_configuration_sha256,
        request.finite_contract.identity_sha256,
        request.context_projection_sha256,
        request.endpoint_definition_sha256,
    )
    observed = (
        spec.reference,
        spec.insight_content_sha256,
        spec.source_evidence_sha256,
        spec.requested_operator_kind,
        spec.source_operator_kinds,
        spec.parent_candidate_id,
        spec.parent_configuration_sha256,
        spec.finite_contract_sha256,
        spec.context_projection_sha256,
        spec.endpoint_definition_sha256,
    )
    if observed != expected:
        raise ValueError("executable hypothesis spec changed trusted request identity")
    if spec.executable_operator_kinds != (request.requested_operator_kind,):
        raise ValueError("compiler broadened executable operator authority")
    immutable_families = tuple(request.insight.recommended_option_families)
    immutable_paths = tuple(sorted(set(request.insight.affected_paths)))
    immutable_predictions = request.insight.effect_predictions
    immutable_falsifier = request.insight.falsification_condition
    if spec.recommended_option_families != immutable_families:
        raise ValueError("compiler changed immutable recommended option families")
    if spec.affected_paths != immutable_paths:
        raise ValueError("compiler changed immutable affected paths")
    if spec.effect_predictions != immutable_predictions:
        raise ValueError("compiler changed immutable metric predictions")
    if spec.falsification_condition != immutable_falsifier:
        raise ValueError("compiler changed immutable falsification condition")
    trusted = {
        TreatmentActionBinding(option.option_id, option.identity_sha256)
        for option in request.finite_contract.options
    }
    if not set(spec.allowed_actions).issubset(trusted):
        raise ValueError("compiler introduced an action outside the finite contract")
    families = {
        request.finite_contract.resolve(action.option_id).family
        for action in spec.allowed_actions
    }
    if not families.issubset(immutable_families):
        raise ValueError("compiler mapped the hypothesis to a foreign action family")
    exact_parent_ids = {
        option_id
        for option_id in request.insight.recommended_option_ids
        if any(
            option.option_id == option_id
            for option in request.finite_contract.options
        )
    }
    if exact_parent_ids and not {
        action.option_id for action in spec.allowed_actions
    }.issubset(exact_parent_ids):
        raise ValueError(
            "compiler ignored immutable exact recommendations valid for this parent"
        )


__all__ = [
    "ExecutableHypothesisTestSpec",
    "HypothesisApplicabilityPort",
    "HypothesisApplicabilityStatus",
    "HypothesisCompilationReceipt",
    "HypothesisCompilationRequest",
    "validate_hypothesis_compiler_identity",
    "validate_hypothesis_compilation",
]
