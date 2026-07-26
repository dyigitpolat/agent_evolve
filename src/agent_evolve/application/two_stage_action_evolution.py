"""Benchmark-inverted execution of the post-reflection AgentEvolve stage.

This module deliberately starts at the first point at which a benchmark has
finished its outcome-blind G1 sample and the reflection workflow has projected
scientific M/P views.  The benchmark supplies those prepared forecast requests,
an identified set utility, and an identified evaluator.  Trusted framework code
then performs the part that must be identical across every problem domain:

* run the M/P/N all-option forecasts concurrently;
* allocate a portfolio independently in each arm while excluding every G1 arm;
* evaluate G2 concurrently under an explicit, receipt-bound reuse policy; and
* synchronously cross an optional-or-required durability/interception boundary
  after every hash-bound phase and before post-decision evaluation authority.

No benchmark metric, configuration schema, evaluator runtime, provider, or
prompt framework is imported here.  A later outer workflow can compose G1 and
strict batched reflection around this service without changing this boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    validate_finite_variation_contract,
    validate_finite_variation_option,
)
from agent_evolve.domain.ids import RunId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ActionAllocationResult,
    DeterministicActionAllocator,
    ForecastPortfolioUtilityBinding,
    validate_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastPolicy,
    ActionForecastRequest,
    ActionForecastResult,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.portfolio_selection import (
    PortfolioExperimentalArm,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_REQUEST_DOMAIN = b"agent-evolve:prepared-two-stage-action-request:v1\x00"
_EVALUATION_REQUEST_DOMAIN = b"agent-evolve:finite-action-evaluation-request:v1\x00"
_EVALUATION_RESULT_DOMAIN = b"agent-evolve:finite-action-evaluation-result:v1\x00"
_PHASE_RECEIPT_DOMAIN = b"agent-evolve:two-stage-phase-receipt:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:prepared-two-stage-action-result:v1\x00"
ACTION_EVALUATION_REUSE_POLICY_ID = "action_evaluation_reuse"
ACTION_EVALUATION_REUSE_POLICY_VERSION = 1
ACTION_EVALUATION_REUSE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:action-evaluation-reuse:v1:"
    b"per_arm=evaluate-each-arm-member-with-no-cross-arm-reuse;"
    b"unique_action=evaluate-each-contract-action-once-and-bind-all-selecting-arms"
).hexdigest()
DURABLE_PHASE_COMMIT_POLICY_ID = "durable_phase_commit"
DURABLE_PHASE_COMMIT_POLICY_VERSION = 1
DURABLE_PHASE_COMMIT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:durable-phase-commit:v1:"
    b"optional=phase-receipts-are-produced-and-a-supplied-sink-must-succeed;"
    b"required=a-sink-must-be-present-and-each-phase-commit-must-complete-before-next-phase;"
    b"allocation-commit-completes-before-evaluator-capability-is-used"
).hexdigest()

SCIENTIFIC_ARM_ORDER = (
    PortfolioExperimentalArm.MEMORY,
    PortfolioExperimentalArm.PERMUTED_PLACEBO,
    PortfolioExperimentalArm.NEUTRAL,
)
_ARM_INDEX = {arm: index for index, arm in enumerate(SCIENTIFIC_ARM_ORDER)}


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


def _agentic_call_telemetry_record(
    telemetry: AgenticCallTelemetry | None,
) -> dict[str, object] | None:
    """Project provider telemetry without lossy Decimal-to-float conversion."""

    if telemetry is None:
        return None
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("telemetry must be exact AgenticCallTelemetry or None")
    telemetry.__post_init__()
    for name in ("provider_response_id", "finish_reason"):
        value = getattr(telemetry, name)
        if value is not None and type(value) is not str:
            raise TypeError(f"telemetry {name} must be an exact string or None")
    if telemetry.cost_usd is not None:
        if type(telemetry.cost_usd) is not Decimal:
            raise TypeError("telemetry cost_usd must be an exact Decimal or None")
        if not telemetry.cost_usd.is_finite():
            raise ValueError("telemetry cost_usd must be finite or None")
        cost_usd: str | None = str(telemetry.cost_usd)
    else:
        cost_usd = None
    return {
        "requested_model": telemetry.requested_model,
        "resolved_model": telemetry.resolved_model,
        "resolved_provider": telemetry.resolved_provider,
        "provider_response_id": telemetry.provider_response_id,
        "finish_reason": telemetry.finish_reason,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "cache_read_tokens": telemetry.cache_read_tokens,
        "cache_write_tokens": telemetry.cache_write_tokens,
        # Decimal is encoded as its exact canonical text, never a binary float.
        "cost_usd": cost_usd,
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


@dataclass(frozen=True, slots=True)
class ActionForecastArmPlan:
    """One fully prepared scientific forecast call."""

    arm: PortfolioExperimentalArm
    request: ActionForecastRequest

    def __post_init__(self) -> None:
        if type(self.arm) is not PortfolioExperimentalArm:
            raise TypeError("arm must be an exact PortfolioExperimentalArm")
        if type(self.request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        self.request.__post_init__()
        receipt = self.request.experimental_view_receipt
        if self.arm is PortfolioExperimentalArm.MEMORY:
            if self.request.evidence_mode is not ActionForecastEvidenceMode.GROUNDED:
                raise ValueError("M must be a grounded forecast request")
            if receipt is None or receipt.arm is not PortfolioExperimentalArm.MEMORY:
                raise ValueError("M must carry a MEMORY experimental-view receipt")
        elif self.arm is PortfolioExperimentalArm.PERMUTED_PLACEBO:
            if self.request.evidence_mode is not ActionForecastEvidenceMode.GROUNDED:
                raise ValueError("P must be a grounded forecast request")
            if (
                receipt is None
                or receipt.arm is not PortfolioExperimentalArm.PERMUTED_PLACEBO
            ):
                raise ValueError(
                    "P must carry a PERMUTED_PLACEBO experimental-view receipt"
                )
        else:
            if self.request.evidence_mode is not ActionForecastEvidenceMode.CATALOG_ONLY:
                raise ValueError("N must be a catalog-only forecast request")
            if receipt is not None:
                raise ValueError("N cannot carry an experimental-view receipt")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"arm": self.arm.value, "request_sha256": self.request.request_sha256}


@runtime_checkable
class FiniteActionEvaluator(Protocol):
    """Benchmark-owned asynchronous evaluation of one sealed child."""

    async def evaluate(
        self,
        request: "FiniteActionEvaluationRequest",
    ) -> FrozenJsonObject: ...


@dataclass(frozen=True, slots=True)
class FiniteActionEvaluatorBinding:
    """Identified benchmark evaluator injected through the application boundary."""

    evaluator: FiniteActionEvaluator = field(repr=False, compare=False)
    evaluator_id: str
    evaluator_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if not callable(getattr(self.evaluator, "evaluate", None)):
            raise TypeError("evaluator must expose an async evaluate method")
        if type(self.evaluator_id) is not str or _TOKEN.fullmatch(
            self.evaluator_id
        ) is None:
            raise ValueError("evaluator_id must use the closed token grammar")
        if type(self.evaluator_version) is not int or self.evaluator_version <= 0:
            raise ValueError("evaluator_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "definition_sha256": self.definition_sha256,
        }


class ActionEvaluationReuseMode(str, Enum):
    """Whether identical G2 actions may share evaluation across study arms."""

    PER_ARM = "per_arm"
    UNIQUE_ACTION = "unique_action"


@dataclass(frozen=True, slots=True)
class ActionEvaluationReusePolicyBinding:
    """Identified compute-accounting policy for cross-arm evaluation reuse."""

    mode: ActionEvaluationReuseMode
    policy_id: str = ACTION_EVALUATION_REUSE_POLICY_ID
    policy_version: int = ACTION_EVALUATION_REUSE_POLICY_VERSION
    definition_sha256: str = ACTION_EVALUATION_REUSE_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.mode) is not ActionEvaluationReuseMode:
            raise TypeError("mode must be an exact ActionEvaluationReuseMode")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "mode": self.mode.value,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


def per_arm_evaluation_reuse_policy() -> ActionEvaluationReusePolicyBinding:
    """Return the fail-safe default: no compute reuse across scientific arms."""

    return ActionEvaluationReusePolicyBinding(ActionEvaluationReuseMode.PER_ARM)


class DurablePhaseCommitRequirement(str, Enum):
    """Whether a run may proceed without a durable phase-commit sink."""

    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class DurablePhaseCommitPolicyBinding:
    """Identified interception policy bound into the scientific run request."""

    requirement: DurablePhaseCommitRequirement
    policy_id: str = DURABLE_PHASE_COMMIT_POLICY_ID
    policy_version: int = DURABLE_PHASE_COMMIT_POLICY_VERSION
    definition_sha256: str = DURABLE_PHASE_COMMIT_POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.requirement) is not DurablePhaseCommitRequirement:
            raise TypeError(
                "requirement must be an exact DurablePhaseCommitRequirement"
            )
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "requirement": self.requirement.value,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
        }


def optional_phase_commit_policy() -> DurablePhaseCommitPolicyBinding:
    """Return the simple-use policy under which a commit sink may be omitted."""

    return DurablePhaseCommitPolicyBinding(DurablePhaseCommitRequirement.OPTIONAL)


def required_scientific_phase_commit_policy() -> DurablePhaseCommitPolicyBinding:
    """Return the fail-closed policy for prospective scientific execution."""

    return DurablePhaseCommitPolicyBinding(DurablePhaseCommitRequirement.REQUIRED)


@dataclass(frozen=True, slots=True)
class PreparedTwoStageActionEvolutionRequest:
    """Prepared G1/reflection outputs and policies for one generic M/P/N run."""

    run_id: RunId
    arm_plans: tuple[ActionForecastArmPlan, ...]
    g1_option_ids: tuple[str, ...]
    portfolio_size: int
    utility: ForecastPortfolioUtilityBinding
    evaluator: FiniteActionEvaluatorBinding
    evaluation_context: FrozenJsonObject
    evaluation_reuse: ActionEvaluationReusePolicyBinding = field(
        default_factory=per_arm_evaluation_reuse_policy
    )
    phase_commit_policy: DurablePhaseCommitPolicyBinding = field(
        default_factory=optional_phase_commit_policy
    )

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId:
            raise TypeError("run_id must be an exact RunId")
        RunId.__post_init__(self.run_id)
        if type(self.arm_plans) is not tuple or any(
            type(value) is not ActionForecastArmPlan for value in self.arm_plans
        ):
            raise TypeError("arm_plans must be an exact ActionForecastArmPlan tuple")
        for plan in self.arm_plans:
            plan.__post_init__()
        if tuple(plan.arm for plan in self.arm_plans) != SCIENTIFIC_ARM_ORDER:
            raise ValueError("arm_plans must contain canonical M/P/N order exactly")
        requests = tuple(plan.request for plan in self.arm_plans)
        baseline = requests[0]
        common = (
            baseline.operation,
            baseline.instruction,
            baseline.context_sha256,
            baseline.optimization_semantics.semantics_id,
            baseline.optimization_semantics.semantics_version,
            baseline.optimization_semantics.definition_sha256,
            baseline.finite_variation_contract.identity_sha256,
            baseline.parent_metric_values,
            baseline.metric_scales,
            baseline.max_output_tokens,
            baseline.temperature,
        )
        for candidate in requests[1:]:
            candidate_common = (
                candidate.operation,
                candidate.instruction,
                candidate.context_sha256,
                candidate.optimization_semantics.semantics_id,
                candidate.optimization_semantics.semantics_version,
                candidate.optimization_semantics.definition_sha256,
                candidate.finite_variation_contract.identity_sha256,
                candidate.parent_metric_values,
                candidate.metric_scales,
                candidate.max_output_tokens,
                candidate.temperature,
            )
            if candidate_common != common:
                raise ValueError(
                    "M/P/N may differ only in call identity and evidence treatment"
                )
        if len({request.call_id for request in requests}) != len(requests):
            raise ValueError("M/P/N require distinct logical call IDs")
        memory_registry = requests[0].source_registry
        placebo_registry = requests[1].source_registry
        assert memory_registry is not None and placebo_registry is not None
        if memory_registry.registry_sha256 != placebo_registry.registry_sha256:
            raise ValueError("M and P must use the same admitted source registry")

        if type(self.g1_option_ids) is not tuple or any(
            type(value) is not str or _OPTION_ID.fullmatch(value) is None
            for value in self.g1_option_ids
        ):
            raise TypeError("g1_option_ids must be an exact option-ID tuple")
        if not self.g1_option_ids:
            raise ValueError("g1_option_ids must be non-empty")
        if self.g1_option_ids != tuple(sorted(set(self.g1_option_ids))):
            raise ValueError("g1_option_ids must be unique and canonical")
        contract = baseline.finite_variation_contract
        validate_finite_variation_contract(contract)
        contract_ids = {option.option_id for option in contract.options}
        if not set(self.g1_option_ids).issubset(contract_ids):
            raise ValueError("g1_option_ids contains an option outside the contract")
        eligible_count = len(contract_ids - set(self.g1_option_ids))
        if type(self.portfolio_size) is not int or self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be a positive exact integer")
        if self.portfolio_size > eligible_count:
            raise ValueError("portfolio_size exceeds the non-G1 action count")
        if type(self.utility) is not ForecastPortfolioUtilityBinding:
            raise TypeError("utility must be an exact identified binding")
        self.utility.__post_init__()
        if type(self.evaluator) is not FiniteActionEvaluatorBinding:
            raise TypeError("evaluator must be an exact identified binding")
        self.evaluator.__post_init__()
        if type(self.evaluation_context) is not FrozenJsonObject:
            raise TypeError("evaluation_context must be an exact FrozenJsonObject")
        if freeze_json(self.evaluation_context) is not self.evaluation_context:
            raise TypeError("evaluation_context must already be frozen typed JSON")
        if type(self.evaluation_reuse) is not ActionEvaluationReusePolicyBinding:
            raise TypeError("evaluation_reuse must be an exact identified binding")
        self.evaluation_reuse.__post_init__()
        if type(self.phase_commit_policy) is not DurablePhaseCommitPolicyBinding:
            raise TypeError("phase_commit_policy must be an exact identified binding")
        self.phase_commit_policy.__post_init__()

    @property
    def finite_variation_contract(self) -> FiniteVariationContract:
        self.__post_init__()
        return self.arm_plans[0].request.finite_variation_contract

    @property
    def eligible_option_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        excluded = set(self.g1_option_ids)
        return tuple(
            sorted(
                option.option_id
                for option in self.finite_variation_contract.options
                if option.option_id not in excluded
            )
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "run_id": self.run_id.value,
            "arm_plans": [plan.to_record() for plan in self.arm_plans],
            "finite_contract_identity_sha256": (
                self.finite_variation_contract.identity_sha256
            ),
            "g1_option_ids": list(self.g1_option_ids),
            "eligible_option_ids": list(self.eligible_option_ids),
            "portfolio_size": self.portfolio_size,
            "utility": self.utility.to_record(),
            "evaluator": self.evaluator.to_record(),
            "evaluation_context_sha256": typed_json_sha256(
                self.evaluation_context
            ),
            "evaluation_reuse": self.evaluation_reuse.to_record(),
            "phase_commit_policy": self.phase_commit_policy.to_record(),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class ActionForecastArmExecution:
    arm: PortfolioExperimentalArm
    request_sha256: str
    result: ActionForecastResult

    def __post_init__(self) -> None:
        if type(self.arm) is not PortfolioExperimentalArm:
            raise TypeError("arm must be exact")
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.result) is not ActionForecastResult:
            raise TypeError("result must be an exact ActionForecastResult")
        self.result.__post_init__()
        if self.result.forecasts.request_sha256 != self.request_sha256:
            raise ValueError("forecast result is bound to a different arm request")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "arm": self.arm.value,
            "request_sha256": self.request_sha256,
            "forecast_receipt_sha256": self.result.forecasts.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActionAllocationArmExecution:
    arm: PortfolioExperimentalArm
    request: ActionAllocationRequest
    result: ActionAllocationResult

    def __post_init__(self) -> None:
        if type(self.arm) is not PortfolioExperimentalArm:
            raise TypeError("arm must be exact")
        if type(self.request) is not ActionAllocationRequest:
            raise TypeError("request must be exact ActionAllocationRequest")
        self.request.__post_init__()
        if type(self.result) is not ActionAllocationResult:
            raise TypeError("result must be exact ActionAllocationResult")
        self.result.__post_init__()
        validate_action_portfolio_decision(self.request, self.result.decision)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "arm": self.arm.value,
            "allocation_request_sha256": self.request.request_sha256,
            "decision_receipt_sha256": self.result.decision.receipt_sha256,
            "selected_option_ids": [
                member.option_id for member in self.result.decision.members
            ],
        }


@dataclass(frozen=True, slots=True)
class FiniteActionEvaluationRequest:
    run_id: RunId
    finite_contract_identity_sha256: str
    option: FiniteVariationOption
    selected_by_arms: tuple[PortfolioExperimentalArm, ...]
    context: FrozenJsonObject

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId:
            raise TypeError("run_id must be exact")
        RunId.__post_init__(self.run_id)
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        validate_finite_variation_option(self.option)
        if type(self.selected_by_arms) is not tuple or not self.selected_by_arms:
            raise ValueError("selected_by_arms must be a non-empty exact tuple")
        if any(type(arm) is not PortfolioExperimentalArm for arm in self.selected_by_arms):
            raise TypeError("selected_by_arms must contain exact arms")
        if self.selected_by_arms != tuple(
            sorted(set(self.selected_by_arms), key=_ARM_INDEX.__getitem__)
        ):
            raise ValueError("selected_by_arms must be unique and canonical")
        if type(self.context) is not FrozenJsonObject:
            raise TypeError("context must be an exact FrozenJsonObject")
        if freeze_json(self.context) is not self.context:
            raise TypeError("context must already be frozen typed JSON")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "run_id": self.run_id.value,
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "option": self.option.evidence_record(),
            "selected_by_arms": [arm.value for arm in self.selected_by_arms],
            "context_sha256": typed_json_sha256(self.context),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_EVALUATION_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class FiniteActionEvaluationResult:
    request: FiniteActionEvaluationRequest
    outcome: FrozenJsonObject
    evaluator_id: str
    evaluator_version: int
    evaluator_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.request) is not FiniteActionEvaluationRequest:
            raise TypeError("request must be an exact FiniteActionEvaluationRequest")
        self.request.__post_init__()
        if type(self.outcome) is not FrozenJsonObject:
            raise TypeError("outcome must be an exact FrozenJsonObject")
        if freeze_json(self.outcome) is not self.outcome:
            raise TypeError("outcome must already be frozen typed JSON")
        if type(self.evaluator_id) is not str or _TOKEN.fullmatch(
            self.evaluator_id
        ) is None:
            raise ValueError("evaluator_id must use the closed token grammar")
        if type(self.evaluator_version) is not int or self.evaluator_version <= 0:
            raise ValueError("evaluator_version must be positive")
        require_sha256(
            self.evaluator_definition_sha256,
            "evaluator_definition_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "evaluation_request_sha256": self.request.request_sha256,
            "option_id": self.request.option.option_id,
            "option_identity_sha256": self.request.option.identity_sha256,
            "child_configuration_sha256": (
                self.request.option.child_configuration_sha256
            ),
            "selected_by_arms": [arm.value for arm in self.request.selected_by_arms],
            "outcome_sha256": typed_json_sha256(self.outcome),
            "evaluator": {
                "evaluator_id": self.evaluator_id,
                "evaluator_version": self.evaluator_version,
                "definition_sha256": self.evaluator_definition_sha256,
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_EVALUATION_RESULT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


class TwoStageActionPhase(str, Enum):
    FORECAST = "forecast"
    ALLOCATE = "allocate"
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class TwoStageActionPhaseReceipt:
    phase: TwoStageActionPhase
    input_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        if type(self.phase) is not TwoStageActionPhase:
            raise TypeError("phase must be an exact TwoStageActionPhase")
        require_sha256(self.input_sha256, "input_sha256")
        require_sha256(self.output_sha256, "output_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "phase": self.phase.value,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_PHASE_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class TwoStageActionPhaseCommit:
    """Exact receipt/payload pair handed to an external durability boundary."""

    receipt: TwoStageActionPhaseReceipt
    payload: FrozenJsonObject

    def __post_init__(self) -> None:
        if type(self.receipt) is not TwoStageActionPhaseReceipt:
            raise TypeError("receipt must be an exact TwoStageActionPhaseReceipt")
        self.receipt.__post_init__()
        if type(self.payload) is not FrozenJsonObject:
            raise TypeError("payload must be an exact FrozenJsonObject")
        if freeze_json(self.payload) is not self.payload:
            raise TypeError("payload must already be frozen typed JSON")
        if typed_json_sha256(self.payload) != self.receipt.output_sha256:
            raise ValueError("phase payload differs from its hash-bound receipt")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "receipt": self.receipt.to_record(),
            "payload_sha256": typed_json_sha256(self.payload),
        }


@runtime_checkable
class TwoStageActionPhaseCommitSink(Protocol):
    """Durably commit one completed phase before the coordinator can continue."""

    def commit(
        self,
        phase_commit: TwoStageActionPhaseCommit,
    ) -> Awaitable[None] | None: ...


class TwoStageActionPhaseCommitError(RuntimeError):
    """A required or supplied phase durability boundary did not complete."""


def _freeze_phase_payload(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:
        raise AssertionError("phase payload root must freeze as an object")
    return frozen


def _phase_commit(
    *,
    receipt: TwoStageActionPhaseReceipt,
    payload: FrozenJsonObject,
) -> TwoStageActionPhaseCommit:
    return TwoStageActionPhaseCommit(receipt=receipt, payload=payload)


async def _publish_phase_commit(
    *,
    policy: DurablePhaseCommitPolicyBinding,
    sink: TwoStageActionPhaseCommitSink | None,
    phase_commit: TwoStageActionPhaseCommit,
) -> None:
    policy.__post_init__()
    phase_commit.__post_init__()
    if sink is None:
        if policy.requirement is DurablePhaseCommitRequirement.REQUIRED:
            raise TwoStageActionPhaseCommitError(
                "required durable phase-commit sink is absent"
            )
        return
    commit_method = getattr(sink, "commit", None)
    if not callable(commit_method):
        raise TypeError("phase_commit_sink must expose a commit method")
    try:
        result = commit_method(phase_commit)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            raise TypeError("phase commit sinks must return None")
    except Exception as exc:
        raise TwoStageActionPhaseCommitError(
            f"{phase_commit.receipt.phase.value} phase commit failed"
        ) from exc


async def _gather_all_settled(
    awaitables: tuple[Awaitable[object], ...],
) -> tuple[object, ...]:
    """Await every sibling and then raise the first failure in input order.

    Scientific multi-arm calls are an all-or-nothing stage.  Waiting for every
    submitted sibling gives the outer artifact recorder a complete physical-
    attempt ledger, while input-order failure selection keeps the observable
    error deterministic.  A failed stage never yields partial results.
    """

    tasks = tuple(asyncio.ensure_future(awaitable) for awaitable in awaitables)
    try:
        results = tuple(
            await asyncio.gather(*tasks, return_exceptions=True)
        )
    except asyncio.CancelledError:
        # Cancellation is not scientific partial success either.  Explicitly
        # cancel and settle every sibling so provider/evaluator cleanup and its
        # physical-attempt journal finish before cancellation escapes.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


@dataclass(frozen=True, slots=True, eq=False)
class PreparedTwoStageActionEvolutionResult:
    request_sha256: str
    forecasts: tuple[ActionForecastArmExecution, ...]
    allocations: tuple[ActionAllocationArmExecution, ...]
    evaluations: tuple[FiniteActionEvaluationResult, ...]
    evaluation_reuse: ActionEvaluationReusePolicyBinding
    phase_commit_policy: DurablePhaseCommitPolicyBinding
    phase_receipts: tuple[TwoStageActionPhaseReceipt, ...]

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if tuple(value.arm for value in self.forecasts) != SCIENTIFIC_ARM_ORDER:
            raise ValueError("forecasts must use canonical M/P/N order")
        if tuple(value.arm for value in self.allocations) != SCIENTIFIC_ARM_ORDER:
            raise ValueError("allocations must use canonical M/P/N order")
        for value in self.forecasts:
            value.__post_init__()
        for value in self.allocations:
            value.__post_init__()
        if type(self.evaluations) is not tuple or not self.evaluations:
            raise ValueError("evaluations must be a non-empty exact tuple")
        for value in self.evaluations:
            if type(value) is not FiniteActionEvaluationResult:
                raise TypeError("evaluations must contain exact results")
            value.__post_init__()
        evaluation_ids = tuple(
            value.request.request_sha256 for value in self.evaluations
        )
        if len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("an exact G2 evaluation request may execute only once")
        if type(self.evaluation_reuse) is not ActionEvaluationReusePolicyBinding:
            raise TypeError("evaluation_reuse must be an exact identified binding")
        self.evaluation_reuse.__post_init__()
        if type(self.phase_commit_policy) is not DurablePhaseCommitPolicyBinding:
            raise TypeError("phase_commit_policy must be an exact identified binding")
        self.phase_commit_policy.__post_init__()
        expected_phases = tuple(TwoStageActionPhase)
        if tuple(value.phase for value in self.phase_receipts) != expected_phases:
            raise ValueError("phase_receipts must use forecast/allocate/evaluate order")
        for value in self.phase_receipts:
            value.__post_init__()

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "forecasts": [value.to_record() for value in self.forecasts],
            "allocations": [value.to_record() for value in self.allocations],
            "evaluations": [value.to_record() for value in self.evaluations],
            "evaluation_reuse": self.evaluation_reuse.to_record(),
            "phase_commit_policy": self.phase_commit_policy.to_record(),
            "phase_receipts": [value.to_record() for value in self.phase_receipts],
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RESULT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PreparedTwoStageActionEvolutionResult
            and type(other) is PreparedTwoStageActionEvolutionResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class PreparedTwoStageActionEvolution:
    """Execute prepared M/P/N forecasts, allocations, and policy-bound G2 evaluation."""

    forecaster: ActionForecastPolicy
    allocator: DeterministicActionAllocator

    def __post_init__(self) -> None:
        if not callable(getattr(self.forecaster, "forecast", None)):
            raise TypeError("forecaster must expose an async forecast method")
        if not callable(getattr(self.allocator, "allocate", None)):
            raise TypeError("allocator must expose an allocate method")

    async def run(
        self,
        request: PreparedTwoStageActionEvolutionRequest,
        *,
        phase_commit_sink: TwoStageActionPhaseCommitSink | None = None,
    ) -> PreparedTwoStageActionEvolutionResult:
        if type(request) is not PreparedTwoStageActionEvolutionRequest:
            raise TypeError("request must be exact PreparedTwoStageActionEvolutionRequest")
        request.__post_init__()
        self.__post_init__()
        if (
            request.phase_commit_policy.requirement
            is DurablePhaseCommitRequirement.REQUIRED
            and phase_commit_sink is None
        ):
            # Fail before spending provider or evaluator compute when a
            # prospective run cannot establish its durability boundary.
            raise TwoStageActionPhaseCommitError(
                "required durable phase-commit sink is absent"
            )
        if phase_commit_sink is not None and not callable(
            getattr(phase_commit_sink, "commit", None)
        ):
            raise TypeError("phase_commit_sink must expose a commit method")

        raw_forecasts = await _gather_all_settled(
            tuple(
                self.forecaster.forecast(plan.request)
                for plan in request.arm_plans
            )
        )
        forecast_executions: list[ActionForecastArmExecution] = []
        for plan, result in zip(request.arm_plans, raw_forecasts, strict=True):
            if type(result) is not ActionForecastResult:
                raise TypeError("forecaster returned a non-ActionForecastResult")
            validate_resolved_action_forecasts(plan.request, result.forecasts)
            forecast_executions.append(
                ActionForecastArmExecution(
                    arm=plan.arm,
                    request_sha256=plan.request.request_sha256,
                    result=result,
                )
            )
        forecasts = tuple(forecast_executions)
        forecast_payload = _freeze_phase_payload(
            {
                "schema_version": 2,
                "phase": TwoStageActionPhase.FORECAST.value,
                "run_request_sha256": request.request_sha256,
                "arm_executions": [
                    {
                        "arm": value.arm.value,
                        "request_sha256": value.request_sha256,
                        "resolved_action_forecast_batch": (
                            value.result.forecasts.to_record()
                        ),
                        "telemetry": _agentic_call_telemetry_record(
                            value.result.telemetry
                        ),
                    }
                    for value in forecasts
                ],
            }
        )
        forecast_receipt = TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.FORECAST,
            input_sha256=_hash(
                b"agent-evolve:two-stage-forecast-input:v1\x00",
                [plan.to_record() for plan in request.arm_plans],
            ),
            output_sha256=typed_json_sha256(forecast_payload),
        )
        await _publish_phase_commit(
            policy=request.phase_commit_policy,
            sink=phase_commit_sink,
            phase_commit=_phase_commit(
                receipt=forecast_receipt,
                payload=forecast_payload,
            ),
        )

        allocations_list: list[ActionAllocationArmExecution] = []
        for plan, forecast in zip(request.arm_plans, forecasts, strict=True):
            allocation_request = ActionAllocationRequest(
                forecast_request=plan.request,
                forecasts=forecast.result.forecasts,
                eligible_option_ids=request.eligible_option_ids,
                portfolio_size=request.portfolio_size,
                utility=request.utility,
            )
            allocation_result = self.allocator.allocate(allocation_request)
            if type(allocation_result) is not ActionAllocationResult:
                raise TypeError("allocator returned a non-ActionAllocationResult")
            allocations_list.append(
                ActionAllocationArmExecution(
                    arm=plan.arm,
                    request=allocation_request,
                    result=allocation_result,
                )
            )
        allocations = tuple(allocations_list)
        allocation_payload = _freeze_phase_payload(
            {
                "schema_version": 1,
                "phase": TwoStageActionPhase.ALLOCATE.value,
                "run_request_sha256": request.request_sha256,
                "arm_executions": [
                    {
                        "arm": value.arm.value,
                        "allocation_request": value.request.to_record(),
                        "decision": value.result.decision.to_record(),
                    }
                    for value in allocations
                ],
            }
        )
        allocation_receipt = TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=forecast_receipt.output_sha256,
            output_sha256=typed_json_sha256(allocation_payload),
        )
        # This await is the oracle-firewall boundary: no evaluator coroutine is
        # even constructed until the selected decisions are durably accepted.
        await _publish_phase_commit(
            policy=request.phase_commit_policy,
            sink=phase_commit_sink,
            phase_commit=_phase_commit(
                receipt=allocation_receipt,
                payload=allocation_payload,
            ),
        )

        if request.evaluation_reuse.mode is ActionEvaluationReuseMode.PER_ARM:
            # Arm order and allocated rank are already canonical.  Repeating the
            # same option across arms intentionally consumes matched compute.
            evaluation_requests = tuple(
                FiniteActionEvaluationRequest(
                    run_id=request.run_id,
                    finite_contract_identity_sha256=(
                        request.finite_variation_contract.identity_sha256
                    ),
                    option=request.finite_variation_contract.resolve(member.option_id),
                    selected_by_arms=(allocation.arm,),
                    context=request.evaluation_context,
                )
                for allocation in allocations
                for member in allocation.result.decision.members
            )
        else:
            selected_by: dict[str, list[PortfolioExperimentalArm]] = {}
            for allocation in allocations:
                for member in allocation.result.decision.members:
                    selected_by.setdefault(member.option_id, []).append(allocation.arm)
            # Contract order, rather than task completion order, is the durable
            # ordering for explicitly reusable deterministic evaluations.
            evaluation_requests = tuple(
                FiniteActionEvaluationRequest(
                    run_id=request.run_id,
                    finite_contract_identity_sha256=(
                        request.finite_variation_contract.identity_sha256
                    ),
                    option=option,
                    selected_by_arms=tuple(selected_by[option.option_id]),
                    context=request.evaluation_context,
                )
                for option in request.finite_variation_contract.options
                if option.option_id in selected_by
            )
        raw_outcomes = await _gather_all_settled(
            tuple(
                request.evaluator.evaluator.evaluate(evaluation_request)
                for evaluation_request in evaluation_requests
            )
        )
        evaluations_list: list[FiniteActionEvaluationResult] = []
        for evaluation_request, outcome in zip(
            evaluation_requests,
            raw_outcomes,
            strict=True,
        ):
            if type(outcome) is not FrozenJsonObject:
                raise TypeError("evaluator returned a non-FrozenJsonObject outcome")
            evaluations_list.append(
                FiniteActionEvaluationResult(
                    request=evaluation_request,
                    outcome=outcome,
                    evaluator_id=request.evaluator.evaluator_id,
                    evaluator_version=request.evaluator.evaluator_version,
                    evaluator_definition_sha256=(
                        request.evaluator.definition_sha256
                    ),
                )
            )
        evaluations = tuple(evaluations_list)
        evaluation_payload = _freeze_phase_payload(
            {
                "schema_version": 1,
                "phase": TwoStageActionPhase.EVALUATE.value,
                "run_request_sha256": request.request_sha256,
                "evaluation_results": [
                    {
                        **value.to_record(),
                        "outcome": thaw_json(value.outcome),
                    }
                    for value in evaluations
                ],
            }
        )
        evaluation_receipt = TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.EVALUATE,
            input_sha256=_hash(
                b"agent-evolve:two-stage-evaluation-input:v1\x00",
                [value.to_record() for value in evaluation_requests],
            ),
            output_sha256=typed_json_sha256(evaluation_payload),
        )
        await _publish_phase_commit(
            policy=request.phase_commit_policy,
            sink=phase_commit_sink,
            phase_commit=_phase_commit(
                receipt=evaluation_receipt,
                payload=evaluation_payload,
            ),
        )
        return PreparedTwoStageActionEvolutionResult(
            request_sha256=request.request_sha256,
            forecasts=forecasts,
            allocations=allocations,
            evaluations=evaluations,
            evaluation_reuse=request.evaluation_reuse,
            phase_commit_policy=request.phase_commit_policy,
            phase_receipts=(
                forecast_receipt,
                allocation_receipt,
                evaluation_receipt,
            ),
        )


__all__ = [
    "ACTION_EVALUATION_REUSE_POLICY_DEFINITION_SHA256",
    "ACTION_EVALUATION_REUSE_POLICY_ID",
    "ACTION_EVALUATION_REUSE_POLICY_VERSION",
    "DURABLE_PHASE_COMMIT_POLICY_DEFINITION_SHA256",
    "DURABLE_PHASE_COMMIT_POLICY_ID",
    "DURABLE_PHASE_COMMIT_POLICY_VERSION",
    "ActionEvaluationReuseMode",
    "ActionEvaluationReusePolicyBinding",
    "ActionAllocationArmExecution",
    "ActionForecastArmExecution",
    "ActionForecastArmPlan",
    "DurablePhaseCommitPolicyBinding",
    "DurablePhaseCommitRequirement",
    "FiniteActionEvaluationRequest",
    "FiniteActionEvaluationResult",
    "FiniteActionEvaluator",
    "FiniteActionEvaluatorBinding",
    "PreparedTwoStageActionEvolution",
    "PreparedTwoStageActionEvolutionRequest",
    "PreparedTwoStageActionEvolutionResult",
    "SCIENTIFIC_ARM_ORDER",
    "TwoStageActionPhase",
    "TwoStageActionPhaseCommit",
    "TwoStageActionPhaseCommitError",
    "TwoStageActionPhaseCommitSink",
    "TwoStageActionPhaseReceipt",
    "optional_phase_commit_policy",
    "per_arm_evaluation_reuse_policy",
    "required_scientific_phase_commit_policy",
]
