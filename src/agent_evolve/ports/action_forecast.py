"""Provider-neutral all-option action forecasts with trusted resolution.

Models author only numeric forecast drafts and opaque citations.  Trusted code
binds a complete draft batch back to one optimization-semantics snapshot, one
parent-bound finite-action contract, and authenticated prompt-visible card
views.  Partial batches are deliberately not representable after resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.core.action_semantics import ActionSpaceSemantics
from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, typed_json_sha256
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioCardSourceRegistry,
    PortfolioExperimentalViewReceipt,
    portfolio_card_action_evidence_sha256,
    portfolio_card_snapshot_sha256,
    validate_portfolio_experimental_view,
)
from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_REQUEST_DOMAIN = b"agent-evolve:action-forecast-request:v2\x00"
_BATCH_DOMAIN = b"agent-evolve:resolved-action-forecast-batch:v2\x00"
_PARTITION_POLICY_DOMAIN = b"agent-evolve:action-forecast-partition-policy:v1\x00"
_BLOCK_SPEC_DOMAIN = b"agent-evolve:action-forecast-block-spec:v1\x00"
_PARTITION_LAYOUT_DOMAIN = b"agent-evolve:action-forecast-partition-layout:v1\x00"
_BLOCK_REQUEST_DOMAIN = b"agent-evolve:action-forecast-block-request:v1\x00"
_RESOLVED_BLOCK_DOMAIN = b"agent-evolve:resolved-action-forecast-block:v1\x00"


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


def _canonical_metric_ids(values: tuple[str, ...], name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _METRIC_ID.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of metric identifiers")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


@dataclass(frozen=True, slots=True)
class ParentMetricValue:
    """One finite parent value in the optimization semantics' metric space."""

    metric_id: str
    value: float

    def __post_init__(self) -> None:
        _canonical_metric_ids((self.metric_id,), "metric_id")
        _finite_float(self.value, "value")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {"metric_id": self.metric_id, "value_hex": self.value.hex()}


@dataclass(frozen=True, slots=True)
class MetricForecastScale:
    """Explicit positive scale used to interpret one metric's delta forecast."""

    metric_id: str
    delta_scale: float
    definition_sha256: str

    def __post_init__(self) -> None:
        _canonical_metric_ids((self.metric_id,), "metric_id")
        _finite_float(self.delta_scale, "delta_scale")
        if self.delta_scale <= 0.0:
            raise ValueError("delta_scale must be strictly positive")
        require_sha256(self.definition_sha256, "definition_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "delta_scale_hex": self.delta_scale.hex(),
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class ActionEvidenceCitation:
    """Model-authored pointer to one prompt-visible card/action binding."""

    card_key: str
    action_binding_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed lowercase token grammar")
        require_sha256(
            self.action_binding_identity_sha256,
            "action_binding_identity_sha256",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "card_key": self.card_key,
            "action_binding_identity_sha256": (
                self.action_binding_identity_sha256
            ),
        }


class ActionForecastEvidenceMode(str, Enum):
    """Request-scoped evidence/citation treatment contract."""

    GROUNDED = "grounded"
    CATALOG_ONLY = "catalog_only"


@dataclass(frozen=True, slots=True)
class ActionMetricForecast:
    """Calibratable delta quantiles for one required metric."""

    metric_id: str
    p10_delta: float
    p50_delta: float
    p90_delta: float
    confidence: float
    citations: tuple[ActionEvidenceCitation, ...]

    def __post_init__(self) -> None:
        _canonical_metric_ids((self.metric_id,), "metric_id")
        for name in ("p10_delta", "p50_delta", "p90_delta", "confidence"):
            _finite_float(getattr(self, name), name)
        if not self.p10_delta <= self.p50_delta <= self.p90_delta:
            raise ValueError("delta quantiles must satisfy p10 <= p50 <= p90")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
        if type(self.citations) is not tuple or any(
            type(value) is not ActionEvidenceCitation for value in self.citations
        ):
            raise TypeError("citations must be an exact tuple of action citations")
        for citation in self.citations:
            citation.__post_init__()
        keys = tuple(
            (value.card_key, value.action_binding_identity_sha256)
            for value in self.citations
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("citations must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "p10_delta_hex": self.p10_delta.hex(),
            "p50_delta_hex": self.p50_delta.hex(),
            "p90_delta_hex": self.p90_delta.hex(),
            "confidence_hex": self.confidence.hex(),
            "citations": [value.to_record() for value in self.citations],
        }


@dataclass(frozen=True, slots=True)
class ActionForecastDraft:
    """Untrusted structured output for exactly one opaque finite option."""

    option_id: str
    probability_valid: float
    metric_forecasts: tuple[ActionMetricForecast, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the closed finite-option grammar")
        _finite_float(self.probability_valid, "probability_valid")
        if not 0.0 <= self.probability_valid <= 1.0:
            raise ValueError("probability_valid must lie in [0,1]")
        if type(self.metric_forecasts) is not tuple or not self.metric_forecasts or any(
            type(value) is not ActionMetricForecast
            for value in self.metric_forecasts
        ):
            raise ValueError(
                "metric_forecasts must be a non-empty exact tuple of forecasts"
            )
        for value in self.metric_forecasts:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.metric_forecasts)
        _canonical_metric_ids(metric_ids, "metric_forecasts")


def _prompt_action_bindings(
    card: PortfolioCard,
) -> tuple[FiniteActionEvidenceBinding, ...]:
    """Read only the authenticated prompt-visible action-evidence projection.

    Source provenance is intentionally not consulted here: experimental card
    views may permute or redact presented action evidence while retaining their
    immutable source binding.  ``PortfolioCard.__post_init__`` authenticates
    whichever projection ``prompt_record`` exposes.
    """

    card.__post_init__()
    return card.finite_action_evidence


def _validate_admitted_card_views(
    cards: tuple[PortfolioCard, ...],
    source_registry: PortfolioCardSourceRegistry,
) -> None:
    """Apply the same closed source/donor checks as scientific selection."""

    if type(source_registry) is not PortfolioCardSourceRegistry:
        raise TypeError("source_registry must be trusted PortfolioCardSourceRegistry")
    source_registry.__post_init__()
    references = tuple(card.reference for card in cards)
    if len(set(references)) != len(references):
        raise ValueError("cards cannot repeat an exact insight reference")
    source_bindings = {
        card.source_binding.binding_sha256: card.source_binding
        for card in cards
        if card.source_binding is not None
    }
    if len(source_bindings) != len(cards):
        raise ValueError("forecast requests require source-bound cards only")
    admitted = {
        binding.binding_sha256: binding
        for binding in source_registry.source_bindings
    }
    if admitted.keys() != source_bindings.keys():
        raise ValueError("source registry differs from the request card source set")
    for binding_sha256, binding in source_bindings.items():
        if admitted[binding_sha256] != binding:
            raise ValueError("source registry binding differs from the request card")
    for card in cards:
        receipt = card.derived_view_receipt
        if receipt is None:
            continue
        evidence_source_sha256 = receipt.evidence_source_binding_sha256
        if evidence_source_sha256 is not None:
            evidence_source = source_bindings.get(evidence_source_sha256)
            if evidence_source is None:
                raise ValueError("derived evidence donor is outside the request")
            if (
                receipt.derived_evidence_sha256
                != evidence_source.source_evidence_sha256
            ):
                raise ValueError("derived evidence differs from its admitted donor")
        score_source_sha256 = receipt.score_source_binding_sha256
        if score_source_sha256 is not None:
            score_source = source_bindings.get(score_source_sha256)
            if score_source is None:
                raise ValueError("derived score donor is outside the request")
            if (
                receipt.derived_score_state_sha256
                != score_source.source_score_state_sha256
            ):
                raise ValueError("derived score state differs from its admitted donor")
        action_source_sha256 = receipt.action_evidence_source_binding_sha256
        if action_source_sha256 is not None:
            action_source = source_bindings.get(action_source_sha256)
            if action_source is None:
                raise ValueError("derived action-evidence donor is outside the request")
            if (
                receipt.derived_action_evidence_sha256
                != portfolio_card_action_evidence_sha256(
                    action_source.finite_action_evidence
                )
            ):
                raise ValueError(
                    "derived action evidence differs from its admitted donor"
                )


@dataclass(frozen=True, slots=True)
class ActionForecastRequest:
    """One logical call that must forecast every option and required metric."""

    call_id: LLMCallId
    operation: str
    instruction: str
    context: FrozenJsonObject
    optimization_semantics: OptimizationSemantics
    action_semantics: ActionSpaceSemantics
    finite_variation_contract: FiniteVariationContract
    cards: tuple[PortfolioCard, ...]
    source_registry: PortfolioCardSourceRegistry | None
    evidence_mode: ActionForecastEvidenceMode
    experimental_view_receipt: PortfolioExperimentalViewReceipt | None
    parent_metric_values: tuple[ParentMetricValue, ...]
    metric_scales: tuple[MetricForecastScale, ...]
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float | None = None

    def __post_init__(self) -> None:
        if type(self.call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.call_id)
        if type(self.operation) is not str or _TOKEN.fullmatch(self.operation) is None:
            raise ValueError("operation must use the closed lowercase token grammar")
        if (
            type(self.instruction) is not str
            or not self.instruction.strip()
            or self.instruction != self.instruction.strip()
        ):
            raise ValueError("instruction must be canonical non-empty text")
        if type(self.context) is not FrozenJsonObject:
            raise TypeError("context must be an exact FrozenJsonObject")
        if freeze_json(self.context) is not self.context:
            raise TypeError("context must already be frozen typed JSON")
        if type(self.optimization_semantics) is not OptimizationSemantics:
            raise TypeError(
                "optimization_semantics must be exact OptimizationSemantics"
            )
        OptimizationSemantics.__post_init__(self.optimization_semantics)
        validate_finite_variation_contract(self.finite_variation_contract)
        if type(self.action_semantics) is not ActionSpaceSemantics:
            raise TypeError("action_semantics must be exact ActionSpaceSemantics")
        ActionSpaceSemantics.__post_init__(self.action_semantics)
        self.action_semantics.validate_contract_binding(
            (
                self.finite_variation_contract.catalog_id,
                self.finite_variation_contract.catalog_version,
                self.finite_variation_contract.catalog_definition_sha256,
            ),
            tuple(
                option.family
                for option in self.finite_variation_contract.options
            ),
        )
        if type(self.cards) is not tuple or any(
            type(card) is not PortfolioCard for card in self.cards
        ):
            raise TypeError("cards must be an exact tuple of PortfolioCard")
        for card in self.cards:
            card.__post_init__()
        card_keys = tuple(card.card_key for card in self.cards)
        if card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("cards must use unique canonical card_key order")
        if type(self.evidence_mode) is not ActionForecastEvidenceMode:
            raise TypeError("evidence_mode must be exact ActionForecastEvidenceMode")
        if self.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
            if not self.cards:
                raise ValueError("grounded forecasts require admitted evidence cards")
            if type(self.source_registry) is not PortfolioCardSourceRegistry:
                raise ValueError("grounded forecasts require a trusted source registry")
            _validate_admitted_card_views(self.cards, self.source_registry)
            if type(self.experimental_view_receipt) is not PortfolioExperimentalViewReceipt:
                raise ValueError(
                    "grounded forecasts require a scientific experimental-view receipt"
                )
            validate_portfolio_experimental_view(
                cards=self.cards,
                finite_variation_contract=self.finite_variation_contract,
                source_registry=self.source_registry,
                receipt=self.experimental_view_receipt,
            )
        else:
            if (
                self.cards
                or self.source_registry is not None
                or self.experimental_view_receipt is not None
            ):
                raise ValueError(
                    "catalog-only forecasts forbid cards, source admission, and "
                    "experimental-view receipts"
                )
        if type(self.parent_metric_values) is not tuple or any(
            type(value) is not ParentMetricValue
            for value in self.parent_metric_values
        ):
            raise TypeError("parent_metric_values must contain exact values")
        if type(self.metric_scales) is not tuple or any(
            type(value) is not MetricForecastScale for value in self.metric_scales
        ):
            raise TypeError("metric_scales must contain exact values")
        for value in self.parent_metric_values:
            value.__post_init__()
        for value in self.metric_scales:
            value.__post_init__()
        parent_ids = tuple(value.metric_id for value in self.parent_metric_values)
        scale_ids = tuple(value.metric_id for value in self.metric_scales)
        _canonical_metric_ids(parent_ids, "parent_metric_values")
        _canonical_metric_ids(scale_ids, "metric_scales")
        if parent_ids != scale_ids:
            raise ValueError("parent metric values and scales must cover identical metrics")
        semantic_ids = {metric.metric_id for metric in self.optimization_semantics.metrics}
        if not set(parent_ids).issubset(semantic_ids):
            raise ValueError("forecast metrics escape the optimization semantics")
        if type(self.max_output_tokens) is not int or not (
            1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"max_output_tokens must lie in [1, {MAX_OUTPUT_TOKENS}]"
            )
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError("temperature must be finite in [0,2] or None")

    @property
    def required_metric_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.metric_id for value in self.parent_metric_values)

    @property
    def context_sha256(self) -> str:
        self.__post_init__()
        return typed_json_sha256(self.context)

    @property
    def card_snapshot_sha256(self) -> str:
        self.__post_init__()
        return portfolio_card_snapshot_sha256(self.cards)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "call_id": self.call_id.value,
            "operation": self.operation,
            "instruction_sha256": hashlib.sha256(
                self.instruction.encode("utf-8", errors="strict")
            ).hexdigest(),
            "context_sha256": self.context_sha256,
            "optimization_semantics": {
                "semantics_id": self.optimization_semantics.semantics_id,
                "semantics_version": self.optimization_semantics.semantics_version,
                "definition_sha256": self.optimization_semantics.definition_sha256,
            },
            "action_semantics": {
                "semantics_id": self.action_semantics.semantics_id,
                "semantics_version": self.action_semantics.semantics_version,
                "definition_sha256": self.action_semantics.definition_sha256,
            },
            "finite_contract_identity_sha256": (
                self.finite_variation_contract.identity_sha256
            ),
            "card_snapshot_sha256": self.card_snapshot_sha256,
            "source_registry_sha256": (
                None
                if self.source_registry is None
                else self.source_registry.registry_sha256
            ),
            "evidence_mode": self.evidence_mode.value,
            "experimental_view_receipt": (
                None
                if self.experimental_view_receipt is None
                else {
                    "arm": self.experimental_view_receipt.arm.value,
                    "receipt_sha256": self.experimental_view_receipt.receipt_sha256,
                }
            ),
            "parent_metric_values": [
                value.to_record() for value in self.parent_metric_values
            ],
            "metric_scales": [value.to_record() for value in self.metric_scales],
            "max_output_tokens": self.max_output_tokens,
            "temperature_hex": (
                None if self.temperature is None else float(self.temperature).hex()
            ),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastPartitionPolicyBinding:
    """Identified, resource-bounded policy for partitioning one target frame."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    max_rows_per_block: int
    max_metric_cells_per_block: int

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if type(self.max_rows_per_block) is not int or self.max_rows_per_block <= 0:
            raise ValueError("max_rows_per_block must be a positive exact integer")
        if (
            type(self.max_metric_cells_per_block) is not int
            or self.max_metric_cells_per_block <= 0
        ):
            raise ValueError(
                "max_metric_cells_per_block must be a positive exact integer"
            )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "max_rows_per_block": self.max_rows_per_block,
            "max_metric_cells_per_block": self.max_metric_cells_per_block,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_PARTITION_POLICY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastPartitionPolicyBinding
            and type(other) is ActionForecastPartitionPolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastBlockSpec:
    """One immutable contiguous slice of global finite-option positions."""

    block_index: int
    global_row_start: int
    global_row_stop: int
    option_identity_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        if type(self.global_row_start) is not int or self.global_row_start < 0:
            raise ValueError(
                "global_row_start must be a non-negative exact integer"
            )
        if (
            type(self.global_row_stop) is not int
            or self.global_row_stop <= self.global_row_start
        ):
            raise ValueError("global_row_stop must be greater than global_row_start")
        if type(self.option_identity_sha256s) is not tuple or any(
            type(value) is not str for value in self.option_identity_sha256s
        ):
            raise TypeError(
                "option_identity_sha256s must be an exact tuple of digests"
            )
        if len(self.option_identity_sha256s) != self.row_count:
            raise ValueError("block option identities must cover every local row")
        for index, value in enumerate(self.option_identity_sha256s):
            require_sha256(value, f"option_identity_sha256s[{index}]")
        if len(set(self.option_identity_sha256s)) != len(
            self.option_identity_sha256s
        ):
            raise ValueError("a block cannot repeat an option identity")

    @property
    def row_count(self) -> int:
        return self.global_row_stop - self.global_row_start

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "block_index": self.block_index,
            "global_row_start": self.global_row_start,
            "global_row_stop": self.global_row_stop,
            "option_identity_sha256s": list(self.option_identity_sha256s),
        }

    @property
    def block_spec_sha256(self) -> str:
        return _hash(_BLOCK_SPEC_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "block_spec_sha256": self.block_spec_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastBlockSpec
            and type(other) is ActionForecastBlockSpec
            and self.block_spec_sha256 == other.block_spec_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastPartitionLayout:
    """Complete deterministic partition of a sealed option/metric frame."""

    finite_contract_identity_sha256: str
    option_identity_sha256s: tuple[str, ...]
    metric_ids: tuple[str, ...]
    partition_policy: ActionForecastPartitionPolicyBinding
    blocks: tuple[ActionForecastBlockSpec, ...]

    def __post_init__(self) -> None:
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        if type(self.option_identity_sha256s) is not tuple or not (
            self.option_identity_sha256s
        ) or any(type(value) is not str for value in self.option_identity_sha256s):
            raise ValueError(
                "option_identity_sha256s must be a non-empty exact tuple"
            )
        for index, value in enumerate(self.option_identity_sha256s):
            require_sha256(value, f"option_identity_sha256s[{index}]")
        if len(set(self.option_identity_sha256s)) != len(
            self.option_identity_sha256s
        ):
            raise ValueError("partition layouts cannot repeat an option identity")
        _canonical_metric_ids(self.metric_ids, "metric_ids")
        if type(self.partition_policy) is not ActionForecastPartitionPolicyBinding:
            raise TypeError(
                "partition_policy must be an exact partition-policy binding"
            )
        self.partition_policy.__post_init__()
        if type(self.blocks) is not tuple or not self.blocks or any(
            type(value) is not ActionForecastBlockSpec for value in self.blocks
        ):
            raise ValueError("blocks must be a non-empty exact tuple of block specs")

        cursor = 0
        metric_count = len(self.metric_ids)
        for expected_index, block in enumerate(self.blocks):
            block.__post_init__()
            if block.block_index != expected_index:
                raise ValueError("block indices must be contiguous canonical positions")
            if block.global_row_start != cursor:
                raise ValueError("blocks must provide gap-free, overlap-free coverage")
            if block.global_row_stop > self.row_count:
                raise ValueError("a block extends beyond the global option frame")
            expected_identities = self.option_identity_sha256s[
                block.global_row_start : block.global_row_stop
            ]
            if block.option_identity_sha256s != expected_identities:
                raise ValueError("block option identities differ from their global slice")
            if block.row_count > self.partition_policy.max_rows_per_block:
                raise ValueError("a block exceeds max_rows_per_block")
            if (
                block.row_count * metric_count
                > self.partition_policy.max_metric_cells_per_block
            ):
                raise ValueError("a block exceeds max_metric_cells_per_block")
            cursor = block.global_row_stop
        if cursor != self.row_count:
            raise ValueError("blocks do not completely cover the global option frame")

    @property
    def row_count(self) -> int:
        return len(self.option_identity_sha256s)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "option_identity_sha256s": list(self.option_identity_sha256s),
            "metric_ids": list(self.metric_ids),
            "partition_policy": self.partition_policy.to_record(),
            "blocks": [value.to_record() for value in self.blocks],
        }

    @property
    def layout_sha256(self) -> str:
        return _hash(_PARTITION_LAYOUT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "layout_sha256": self.layout_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastPartitionLayout
            and type(other) is ActionForecastPartitionLayout
            and self.layout_sha256 == other.layout_sha256
        )

    __hash__ = None


def validate_action_forecast_partition_layout(
    request: ActionForecastRequest,
    layout: ActionForecastPartitionLayout,
) -> None:
    """Bind a reusable partition geometry to one exact forecast request."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    if type(layout) is not ActionForecastPartitionLayout:
        raise TypeError("layout must be an exact ActionForecastPartitionLayout")
    layout.__post_init__()
    contract = request.finite_variation_contract
    if layout.finite_contract_identity_sha256 != contract.identity_sha256:
        raise ValueError("partition layout is bound to a different finite contract")
    if layout.option_identity_sha256s != tuple(
        option.identity_sha256 for option in contract.options
    ):
        raise ValueError("partition layout option frame differs from the request")
    if layout.metric_ids != request.required_metric_ids:
        raise ValueError("partition layout metric frame differs from the request")


@dataclass(frozen=True, slots=True, eq=False)
class ActionForecastBlockRequest:
    """One exact provider-neutral block call within a global logical request."""

    request: ActionForecastRequest
    layout: ActionForecastPartitionLayout
    block: ActionForecastBlockSpec
    block_call_id: LLMCallId

    def __post_init__(self) -> None:
        if type(self.request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        self.request.__post_init__()
        validate_action_forecast_partition_layout(self.request, self.layout)
        if type(self.block) is not ActionForecastBlockSpec:
            raise TypeError("block must be an exact ActionForecastBlockSpec")
        self.block.__post_init__()
        if self.block.block_index >= self.layout.block_count:
            raise ValueError("block index is outside the partition layout")
        if self.block != self.layout.blocks[self.block.block_index]:
            raise ValueError("block differs from its exact partition-layout position")
        if type(self.block_call_id) is not LLMCallId:
            raise TypeError("block_call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.block_call_id)

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request.request_sha256,
            "layout_sha256": self.layout.layout_sha256,
            "block_spec_sha256": self.block.block_spec_sha256,
            "block_index": self.block.block_index,
            "block_call_id": self.block_call_id.value,
        }

    @property
    def block_request_sha256(self) -> str:
        return _hash(_BLOCK_REQUEST_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "block_request_sha256": self.block_request_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionForecastBlockRequest
            and type(other) is ActionForecastBlockRequest
            and self.block_request_sha256 == other.block_request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class ResolvedActionEvidenceCitation:
    """Trusted citation resolved against the exact prompt-visible card view."""

    card_key: str
    card_source_binding_sha256: str
    action_binding_identity_sha256: str
    contrast_id: str
    source_option_id: str
    source_option_identity_sha256: str
    source_contract_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed lowercase token grammar")
        if type(self.source_option_id) is not str or _OPTION_ID.fullmatch(
            self.source_option_id
        ) is None:
            raise ValueError("source_option_id must use the finite-option grammar")
        for name in (
            "card_source_binding_sha256",
            "action_binding_identity_sha256",
            "contrast_id",
            "source_option_identity_sha256",
            "source_contract_identity_sha256",
        ):
            require_sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "card_key": self.card_key,
            "card_source_binding_sha256": self.card_source_binding_sha256,
            "action_binding_identity_sha256": self.action_binding_identity_sha256,
            "contrast_id": self.contrast_id,
            "source_option_id": self.source_option_id,
            "source_option_identity_sha256": self.source_option_identity_sha256,
            "source_contract_identity_sha256": self.source_contract_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResolvedActionMetricForecast:
    metric_id: str
    p10_delta: float
    p50_delta: float
    p90_delta: float
    confidence: float
    citations: tuple[ResolvedActionEvidenceCitation, ...]

    def __post_init__(self) -> None:
        if type(self.citations) is not tuple or any(
            type(value) is not ResolvedActionEvidenceCitation
            for value in self.citations
        ):
            raise TypeError("citations must be an exact tuple of resolved citations")
        for value in self.citations:
            value.__post_init__()
        unresolved = ActionMetricForecast(
            metric_id=self.metric_id,
            p10_delta=self.p10_delta,
            p50_delta=self.p50_delta,
            p90_delta=self.p90_delta,
            confidence=self.confidence,
            citations=tuple(
                ActionEvidenceCitation(
                    card_key=value.card_key,
                    action_binding_identity_sha256=(
                        value.action_binding_identity_sha256
                    ),
                )
                for value in self.citations
            ),
        )
        del unresolved

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "p10_delta_hex": self.p10_delta.hex(),
            "p50_delta_hex": self.p50_delta.hex(),
            "p90_delta_hex": self.p90_delta.hex(),
            "confidence_hex": self.confidence.hex(),
            "citations": [value.to_record() for value in self.citations],
        }


@dataclass(frozen=True, slots=True)
class ResolvedActionForecast:
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str
    probability_valid: float
    metric_forecasts: tuple[ResolvedActionMetricForecast, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(self.option_id) is None:
            raise ValueError("option_id must use the finite-option grammar")
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the closed token grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )
        _finite_float(self.probability_valid, "probability_valid")
        if not 0.0 <= self.probability_valid <= 1.0:
            raise ValueError("probability_valid must lie in [0,1]")
        if type(self.metric_forecasts) is not tuple or not self.metric_forecasts or any(
            type(value) is not ResolvedActionMetricForecast
            for value in self.metric_forecasts
        ):
            raise ValueError("metric_forecasts must contain resolved values")
        for value in self.metric_forecasts:
            value.__post_init__()
        _canonical_metric_ids(
            tuple(value.metric_id for value in self.metric_forecasts),
            "metric_forecasts",
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
            "probability_valid_hex": self.probability_valid.hex(),
            "metric_forecasts": [
                value.to_record() for value in self.metric_forecasts
            ],
        }


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedActionForecastBlock:
    """Trusted partial forecast receipt for one exact partition block."""

    request_sha256: str
    block_request_sha256: str
    layout_sha256: str
    block_spec_sha256: str
    block_index: int
    forecasts: tuple[ResolvedActionForecast, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "block_request_sha256",
            "layout_sha256",
            "block_spec_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        if type(self.forecasts) is not tuple or not self.forecasts or any(
            type(value) is not ResolvedActionForecast for value in self.forecasts
        ):
            raise ValueError("forecasts must be a non-empty exact resolved tuple")
        for value in self.forecasts:
            value.__post_init__()
        option_ids = tuple(value.option_id for value in self.forecasts)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("a resolved block cannot repeat an option")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "block_request_sha256": self.block_request_sha256,
            "layout_sha256": self.layout_sha256,
            "block_spec_sha256": self.block_spec_sha256,
            "block_index": self.block_index,
            "forecasts": [value.to_record() for value in self.forecasts],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_RESOLVED_BLOCK_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ResolvedActionForecastBlock
            and type(other) is ResolvedActionForecastBlock
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedActionForecastBatch:
    """Complete all-option forecast receipt bound to one exact request."""

    request_sha256: str
    context_sha256: str
    optimization_semantics_definition_sha256: str
    action_semantics_definition_sha256: str
    finite_contract_identity_sha256: str
    card_snapshot_sha256: str
    forecasts: tuple[ResolvedActionForecast, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "context_sha256",
            "optimization_semantics_definition_sha256",
            "action_semantics_definition_sha256",
            "finite_contract_identity_sha256",
            "card_snapshot_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.forecasts) is not tuple or not self.forecasts or any(
            type(value) is not ResolvedActionForecast for value in self.forecasts
        ):
            raise ValueError("forecasts must be a non-empty exact resolved tuple")
        for value in self.forecasts:
            value.__post_init__()
        option_ids = tuple(value.option_id for value in self.forecasts)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("resolved forecasts cannot repeat an option")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "request_sha256": self.request_sha256,
            "context_sha256": self.context_sha256,
            "optimization_semantics_definition_sha256": (
                self.optimization_semantics_definition_sha256
            ),
            "action_semantics_definition_sha256": (
                self.action_semantics_definition_sha256
            ),
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "card_snapshot_sha256": self.card_snapshot_sha256,
            "forecasts": [value.to_record() for value in self.forecasts],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_BATCH_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ResolvedActionForecastBatch
            and type(other) is ResolvedActionForecastBatch
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def _resolved_citation_index(
    request: ActionForecastRequest,
) -> dict[tuple[str, str], ResolvedActionEvidenceCitation]:
    """Resolve each immutable prompt-visible citation exactly once.

    Callers validate the complete request before constructing this index.  The
    request and every nested card/binding are frozen values, so rebuilding and
    revalidating the same provenance graph for every option/metric citation
    adds no protection.  Keeping the exact card key in the lookup key preserves
    the original rejection of cross-card and ambiguous binding references.
    """

    index: dict[tuple[str, str], ResolvedActionEvidenceCitation] = {}
    for card in request.cards:
        if card.source_binding is None:
            raise ValueError("forecast request contains an unbound evidence card")
        for binding in _prompt_action_bindings(card):
            key = (card.card_key, binding.identity_sha256)
            if key in index:
                raise ValueError(
                    "forecast request contains an ambiguous prompt action binding"
                )
            index[key] = ResolvedActionEvidenceCitation(
                card_key=card.card_key,
                card_source_binding_sha256=card.source_binding.binding_sha256,
                action_binding_identity_sha256=binding.identity_sha256,
                contrast_id=binding.contrast_id,
                source_option_id=binding.option_id,
                source_option_identity_sha256=binding.option_identity_sha256,
                source_contract_identity_sha256=binding.contract_identity_sha256,
            )
    return index


def _resolve_indexed_citation(
    citation_index: dict[tuple[str, str], ResolvedActionEvidenceCitation],
    *,
    prompt_card_keys: frozenset[str],
    citation: ActionEvidenceCitation,
) -> ResolvedActionEvidenceCitation:
    if citation.card_key not in prompt_card_keys:
        raise ValueError("forecast cites a card outside the request snapshot")
    resolved = citation_index.get(
        (citation.card_key, citation.action_binding_identity_sha256)
    )
    if resolved is None:
        raise ValueError("forecast cites an absent or ambiguous prompt action binding")
    return resolved


def resolve_action_forecast_block(
    block_request: ActionForecastBlockRequest,
    drafts: tuple[ActionForecastDraft, ...],
    *,
    policy_id: str,
    policy_version: int,
    policy_definition_sha256: str,
) -> ResolvedActionForecastBlock:
    """Resolve exactly one block without representing it as a complete batch."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be an exact ActionForecastBlockRequest")
    block_request.__post_init__()
    request = block_request.request
    block_spec = block_request.block
    options = request.finite_variation_contract.options[
        block_spec.global_row_start : block_spec.global_row_stop
    ]
    if type(drafts) is not tuple or any(
        type(value) is not ActionForecastDraft for value in drafts
    ):
        raise TypeError("drafts must be an exact tuple of ActionForecastDraft")
    for value in drafts:
        value.__post_init__()
    if tuple(value.option_id for value in drafts) != tuple(
        option.option_id for option in options
    ):
        raise ValueError(
            "block forecast order/coverage differs from its global option slice"
        )

    required_metric_ids = request.required_metric_ids
    citation_index = _resolved_citation_index(request)
    prompt_card_keys = frozenset(card.card_key for card in request.cards)
    resolved: list[ResolvedActionForecast] = []
    for option, draft in zip(options, drafts, strict=True):
        if tuple(value.metric_id for value in draft.metric_forecasts) != (
            required_metric_ids
        ):
            raise ValueError("every block row must forecast every required metric")
        for metric in draft.metric_forecasts:
            if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
                if not metric.citations:
                    raise ValueError(
                        "grounded block forecasts require citations for every metric"
                    )
            elif metric.citations:
                raise ValueError("catalog-only block forecasts forbid citations")
        resolved.append(
            ResolvedActionForecast(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                family=option.family,
                probability_valid=draft.probability_valid,
                metric_forecasts=tuple(
                    ResolvedActionMetricForecast(
                        metric_id=metric.metric_id,
                        p10_delta=metric.p10_delta,
                        p50_delta=metric.p50_delta,
                        p90_delta=metric.p90_delta,
                        confidence=metric.confidence,
                        citations=tuple(
                            _resolve_indexed_citation(
                                citation_index,
                                prompt_card_keys=prompt_card_keys,
                                citation=citation,
                            )
                            for citation in metric.citations
                        ),
                    )
                    for metric in draft.metric_forecasts
                ),
            )
        )
    result = ResolvedActionForecastBlock(
        request_sha256=request.request_sha256,
        block_request_sha256=block_request.block_request_sha256,
        layout_sha256=block_request.layout.layout_sha256,
        block_spec_sha256=block_spec.block_spec_sha256,
        block_index=block_spec.block_index,
        forecasts=tuple(resolved),
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
    )
    validate_resolved_action_forecast_block(block_request, result)
    return result


def validate_resolved_action_forecast_block(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
) -> None:
    """Revalidate one resolved partial receipt against its global positions."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be an exact ActionForecastBlockRequest")
    block_request.__post_init__()
    if type(block) is not ResolvedActionForecastBlock:
        raise TypeError("block must be an exact ResolvedActionForecastBlock")
    block.__post_init__()
    request = block_request.request
    spec = block_request.block
    if (
        block.request_sha256 != request.request_sha256
        or block.block_request_sha256 != block_request.block_request_sha256
        or block.layout_sha256 != block_request.layout.layout_sha256
        or block.block_spec_sha256 != spec.block_spec_sha256
        or block.block_index != spec.block_index
    ):
        raise ValueError("resolved block is bound to a different block request")
    options = request.finite_variation_contract.options[
        spec.global_row_start : spec.global_row_stop
    ]
    if tuple(value.option_id for value in block.forecasts) != tuple(
        option.option_id for option in options
    ):
        raise ValueError("resolved block order/coverage differs from its global slice")

    required_metric_ids = request.required_metric_ids
    citation_index = _resolved_citation_index(request)
    prompt_card_keys = frozenset(card.card_key for card in request.cards)
    for option, forecast in zip(options, block.forecasts, strict=True):
        if (
            forecast.option_identity_sha256 != option.identity_sha256
            or forecast.child_configuration_sha256
            != option.child_configuration_sha256
            or forecast.family != option.family
        ):
            raise ValueError("resolved block differs from its sealed finite option")
        if tuple(value.metric_id for value in forecast.metric_forecasts) != (
            required_metric_ids
        ):
            raise ValueError("resolved block metric coverage differs from request")
        for metric in forecast.metric_forecasts:
            if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
                if not metric.citations:
                    raise ValueError(
                        "grounded resolved blocks require citations for every metric"
                    )
            elif metric.citations:
                raise ValueError("catalog-only resolved blocks forbid citations")
            for citation in metric.citations:
                unresolved = ActionEvidenceCitation(
                    card_key=citation.card_key,
                    action_binding_identity_sha256=(
                        citation.action_binding_identity_sha256
                    ),
                )
                expected = _resolve_indexed_citation(
                    citation_index,
                    prompt_card_keys=prompt_card_keys,
                    citation=unresolved,
                )
                if citation != expected:
                    raise ValueError(
                        "resolved block citation differs from prompt card evidence"
                    )


def resolve_action_forecasts(
    request: ActionForecastRequest,
    drafts: tuple[ActionForecastDraft, ...],
    *,
    policy_id: str,
    policy_version: int,
    policy_definition_sha256: str,
) -> ResolvedActionForecastBatch:
    """Resolve one complete all-option output or publish no forecast receipt."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    required_metric_ids = tuple(
        value.metric_id for value in request.parent_metric_values
    )
    request_sha256 = request.request_sha256
    context_sha256 = typed_json_sha256(request.context)
    contract = request.finite_variation_contract
    contract_identity_sha256 = contract.identity_sha256
    card_snapshot_sha256 = portfolio_card_snapshot_sha256(request.cards)
    if type(drafts) is not tuple or any(
        type(value) is not ActionForecastDraft for value in drafts
    ):
        raise TypeError("drafts must be an exact tuple of ActionForecastDraft")
    for value in drafts:
        value.__post_init__()
    option_ids = tuple(value.option_id for value in drafts)
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("forecast output repeats a finite option")
    expected = {option.option_id for option in contract.options}
    foreign = set(option_ids) - expected
    if foreign:
        raise ValueError("forecast output contains a foreign finite option")
    missing = expected - set(option_ids)
    if missing:
        raise ValueError("forecast output is incomplete for the finite contract")
    by_option = {value.option_id: value for value in drafts}
    citation_index = _resolved_citation_index(request)
    prompt_card_keys = frozenset(card.card_key for card in request.cards)
    resolved: list[ResolvedActionForecast] = []
    for option in contract.options:
        draft = by_option[option.option_id]
        if tuple(value.metric_id for value in draft.metric_forecasts) != (
            required_metric_ids
        ):
            raise ValueError("every option must forecast every required metric exactly")
        for metric in draft.metric_forecasts:
            if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
                if not metric.citations:
                    raise ValueError(
                        "grounded forecasts require card/action citations for every metric"
                    )
            elif metric.citations:
                raise ValueError("catalog-only forecasts forbid evidence citations")
        resolved.append(
            ResolvedActionForecast(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                family=option.family,
                probability_valid=draft.probability_valid,
                metric_forecasts=tuple(
                    ResolvedActionMetricForecast(
                        metric_id=metric.metric_id,
                        p10_delta=metric.p10_delta,
                        p50_delta=metric.p50_delta,
                        p90_delta=metric.p90_delta,
                        confidence=metric.confidence,
                        citations=tuple(
                            _resolve_indexed_citation(
                                citation_index,
                                prompt_card_keys=prompt_card_keys,
                                citation=citation,
                            )
                            for citation in metric.citations
                        ),
                    )
                    for metric in draft.metric_forecasts
                ),
            )
        )
    batch = ResolvedActionForecastBatch(
        request_sha256=request_sha256,
        context_sha256=context_sha256,
        optimization_semantics_definition_sha256=(
            request.optimization_semantics.definition_sha256
        ),
        action_semantics_definition_sha256=(
            request.action_semantics.definition_sha256
        ),
        finite_contract_identity_sha256=contract_identity_sha256,
        card_snapshot_sha256=card_snapshot_sha256,
        forecasts=tuple(resolved),
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
    )
    validate_resolved_action_forecasts(request, batch)
    return batch


def validate_resolved_action_forecasts(
    request: ActionForecastRequest,
    batch: ResolvedActionForecastBatch,
) -> None:
    """Revalidate a forecast receipt against its exact trusted request."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    required_metric_ids = tuple(
        value.metric_id for value in request.parent_metric_values
    )
    request_sha256 = request.request_sha256
    context_sha256 = typed_json_sha256(request.context)
    contract = request.finite_variation_contract
    contract_identity_sha256 = contract.identity_sha256
    card_snapshot_sha256 = portfolio_card_snapshot_sha256(request.cards)
    options_by_id = {option.option_id: option for option in contract.options}
    if type(batch) is not ResolvedActionForecastBatch:
        raise TypeError("batch must be an exact ResolvedActionForecastBatch")
    batch.__post_init__()
    if (
        batch.request_sha256 != request_sha256
        or batch.context_sha256 != context_sha256
        or batch.optimization_semantics_definition_sha256
        != request.optimization_semantics.definition_sha256
        or batch.action_semantics_definition_sha256
        != request.action_semantics.definition_sha256
        or batch.finite_contract_identity_sha256
        != contract_identity_sha256
        or batch.card_snapshot_sha256 != card_snapshot_sha256
    ):
        raise ValueError("forecast batch is bound to a different request snapshot")
    if tuple(value.option_id for value in batch.forecasts) != tuple(
        option.option_id for option in contract.options
    ):
        raise ValueError("resolved forecast order/coverage differs from the contract")
    citation_index = _resolved_citation_index(request)
    prompt_card_keys = frozenset(card.card_key for card in request.cards)
    for forecast in batch.forecasts:
        option = options_by_id.get(forecast.option_id)
        if option is None:
            raise ValueError("resolved forecast option is outside the sealed contract")
        if (
            forecast.option_identity_sha256 != option.identity_sha256
            or forecast.child_configuration_sha256
            != option.child_configuration_sha256
            or forecast.family != option.family
        ):
            raise ValueError("resolved forecast differs from its sealed finite option")
        if tuple(value.metric_id for value in forecast.metric_forecasts) != (
            required_metric_ids
        ):
            raise ValueError("resolved forecast metric coverage differs from request")
        for metric in forecast.metric_forecasts:
            if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
                if not metric.citations:
                    raise ValueError(
                        "grounded resolved forecasts require card/action "
                        "citations for every metric"
                    )
            elif metric.citations:
                raise ValueError(
                    "catalog-only resolved forecasts forbid evidence citations"
                )
            for citation in metric.citations:
                unresolved = ActionEvidenceCitation(
                    card_key=citation.card_key,
                    action_binding_identity_sha256=(
                        citation.action_binding_identity_sha256
                    ),
                )
                expected = _resolve_indexed_citation(
                    citation_index,
                    prompt_card_keys=prompt_card_keys,
                    citation=unresolved,
                )
                if citation != expected:
                    raise ValueError("resolved citation differs from prompt card evidence")


@dataclass(frozen=True, slots=True)
class ActionForecastResult:
    forecasts: ResolvedActionForecastBatch
    telemetry: AgenticCallTelemetry | None

    def __post_init__(self) -> None:
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be an exact ResolvedActionForecastBatch")
        self.forecasts.__post_init__()
        if self.telemetry is not None:
            if type(self.telemetry) is not AgenticCallTelemetry:
                raise TypeError("telemetry must be exact AgenticCallTelemetry or None")
            self.telemetry.__post_init__()


@dataclass(frozen=True, slots=True)
class ActionForecastBlockResult:
    """One physical block execution and its optional provider telemetry."""

    forecasts: ResolvedActionForecastBlock
    telemetry: AgenticCallTelemetry | None

    def __post_init__(self) -> None:
        if type(self.forecasts) is not ResolvedActionForecastBlock:
            raise TypeError(
                "forecasts must be an exact ResolvedActionForecastBlock"
            )
        self.forecasts.__post_init__()
        if self.telemetry is not None:
            if type(self.telemetry) is not AgenticCallTelemetry:
                raise TypeError("telemetry must be exact AgenticCallTelemetry or None")
            self.telemetry.__post_init__()


@runtime_checkable
class ActionForecastPolicy(Protocol):
    """Forecast every sealed option; partial provider results cannot escape."""

    async def forecast(self, request: ActionForecastRequest) -> ActionForecastResult: ...


@runtime_checkable
class ActionForecastBlockPolicy(Protocol):
    """Forecast one sealed block; the application owns global reassembly."""

    async def forecast_block(
        self,
        request: ActionForecastBlockRequest,
    ) -> ActionForecastBlockResult: ...


__all__ = [
    "ActionEvidenceCitation",
    "ActionForecastBlockPolicy",
    "ActionForecastBlockRequest",
    "ActionForecastBlockResult",
    "ActionForecastBlockSpec",
    "ActionForecastDraft",
    "ActionForecastEvidenceMode",
    "ActionForecastPartitionLayout",
    "ActionForecastPartitionPolicyBinding",
    "ActionForecastPolicy",
    "ActionForecastRequest",
    "ActionForecastResult",
    "ActionMetricForecast",
    "MetricForecastScale",
    "ParentMetricValue",
    "ResolvedActionEvidenceCitation",
    "ResolvedActionForecast",
    "ResolvedActionForecastBatch",
    "ResolvedActionForecastBlock",
    "ResolvedActionMetricForecast",
    "resolve_action_forecast_block",
    "resolve_action_forecasts",
    "validate_action_forecast_partition_layout",
    "validate_resolved_action_forecast_block",
    "validate_resolved_action_forecasts",
]
