"""High-level agentic generation port used by the evolutionary workflow."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, Tuple, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    require_sha256,
    validate_json_path,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_json_scalar,
    typed_json_equal,
)


_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINITE_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_REFLECTION_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_FINITE_OPTION_FAMILY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_REFLECTION_INSIGHT_CONTRACT_DOMAIN = (
    b"agent-evolve:reflection-insight-contract:v2\x00"
)
_INSIGHT_DRAFT_CONTENT_DOMAIN = b"agent-evolve:insight-draft-content:v1\x00"


def _validate_contrast_ids(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _LOWER_SHA256.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of lowercase SHA-256 IDs")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _validate_canonical_tokens(
    values: tuple[str, ...],
    *,
    name: str,
    pattern: re.Pattern[str],
) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or pattern.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of canonical identifiers")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


class MetricEffectDirection(str, Enum):
    """Closed, testable prediction for one explicitly named numeric metric."""

    DECREASE = "decrease"
    INCREASE = "increase"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MetricEffectPrediction:
    """One direction prediction, independent of any domain's goal semantics."""

    metric_id: str
    direction: MetricEffectDirection

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _REFLECTION_METRIC_ID.fullmatch(
            self.metric_id
        ) is None:
            raise ValueError(
                "metric_id must use the closed reflection identifier grammar"
            )
        if type(self.direction) is not MetricEffectDirection:
            raise TypeError("direction must be an exact MetricEffectDirection")


@dataclass(frozen=True, slots=True)
class SourceAttribution:
    path: str
    source: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path.strip():
            raise ValueError("attribution path must be non-empty")
        if self.source not in {"ancestor", "left", "right", "synthesized", "mutation"}:
            raise ValueError("unsupported attribution source")


@dataclass(frozen=True, slots=True)
class ConflictResolutionDraft:
    relation_id: str
    choice: str
    explanation: str

    def __post_init__(self) -> None:
        if type(self.relation_id) is not str or not self.relation_id.strip():
            raise ValueError("relation_id must be non-empty")
        if self.choice not in {"choose_left", "choose_right", "synthesize", "drop_both"}:
            raise ValueError("unsupported conflict resolution choice")
        if type(self.explanation) is not str or not self.explanation.strip():
            raise ValueError("resolution explanation must be non-empty")


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    configuration: dict[str, Any]
    design_rationale: str
    intended_changes: Tuple[str, ...] = ()
    source_attribution: Tuple[SourceAttribution, ...] = ()
    claimed_insight_ids: Tuple[str, ...] = ()
    claimed_preservation_obligation_ids: Tuple[str, ...] = ()
    conflict_resolutions: Tuple[ConflictResolutionDraft, ...] = ()

    def __post_init__(self) -> None:
        if type(self.configuration) is not dict:
            raise TypeError("configuration must be an exact dict")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        for name in (
            "intended_changes",
            "claimed_insight_ids",
            "claimed_preservation_obligation_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise TypeError(f"{name} must be an exact tuple of non-empty strings")
        if type(self.source_attribution) is not tuple or any(
            type(value) is not SourceAttribution for value in self.source_attribution
        ):
            raise TypeError("source_attribution must contain exact values")
        if type(self.conflict_resolutions) is not tuple or any(
            type(value) is not ConflictResolutionDraft
            for value in self.conflict_resolutions
        ):
            raise TypeError("conflict_resolutions must contain exact values")


def _frozen_value_at_path(
    value: FrozenJsonValue,
    path: JsonPath,
) -> FrozenJsonValue:
    """Resolve an already-frozen value without introducing a policy dependency."""

    current = value
    for segment in path.segments:
        if type(segment) is ObjectKey:
            if type(current) is not FrozenJsonObject:
                raise ValueError("atomic mutation path reaches a non-object")
            matches = tuple(
                item for key, item in current.items if key == segment.value
            )
            if len(matches) != 1:
                raise ValueError("atomic mutation path does not exist")
            current = matches[0]
        elif type(segment) is ArrayIndex:
            if type(current) is not FrozenJsonArray:
                raise ValueError("atomic mutation path reaches a non-array")
            if segment.value >= len(current.items):
                raise ValueError("atomic mutation path is out of bounds")
            current = current.items[segment.value]
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported path segment")
    return current


@dataclass(frozen=True, slots=True)
class AtomicMutationOutputContract:
    """Immutable parent and exact scalar leaf exposed to one patch-native call."""

    parent_configuration: FrozenJsonObject
    editable_path: JsonPath
    replacement_options: Tuple[FrozenJsonValue, ...] = ()

    def __post_init__(self) -> None:
        if type(self.parent_configuration) is not FrozenJsonObject:
            raise TypeError("parent_configuration must be an exact FrozenJsonObject")
        if freeze_json(self.parent_configuration) is not self.parent_configuration:
            raise TypeError("parent_configuration must already be frozen typed JSON")
        if type(self.editable_path) is not JsonPath:
            raise TypeError("editable_path must be an exact JsonPath")
        validate_json_path(self.editable_path)
        if not self.editable_path.segments:
            raise ValueError("atomic mutation cannot replace the candidate root")
        current = _frozen_value_at_path(
            self.parent_configuration,
            self.editable_path,
        )
        if not is_json_scalar(current):
            raise ValueError("atomic mutation editable_path must resolve to a scalar")
        if type(self.replacement_options) is not tuple:
            raise TypeError("replacement_options must be an exact tuple")
        canonical_options: list[bytes] = []
        for option in self.replacement_options:
            if freeze_json(option) is not option or not is_json_scalar(option):
                raise TypeError(
                    "replacement_options must contain frozen typed-JSON scalars"
                )
            if typed_json_equal(current, option):
                raise ValueError("replacement_options must exclude the parent value")
            canonical_options.append(canonical_typed_json_bytes(option))
        if len(set(canonical_options)) != len(canonical_options):
            raise ValueError("replacement_options cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class AtomicMutationDraft:
    """One model-authored scalar replacement; the engine owns materialization."""

    path: JsonPath
    replacement: FrozenJsonValue
    design_rationale: str
    claimed_insight_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.path) is not JsonPath:
            raise TypeError("path must be an exact JsonPath")
        validate_json_path(self.path)
        if not self.path.segments:
            raise ValueError("an atomic mutation path cannot be the root")
        frozen_replacement = freeze_json(self.replacement)
        if frozen_replacement is not self.replacement:
            raise TypeError("replacement must already be frozen typed JSON")
        if not is_json_scalar(self.replacement):
            raise TypeError("replacement must be an exact typed-JSON scalar")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must be an exact tuple of non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class FiniteVariationSelectionDraft:
    """A model-selected ID bound to the exact presealed option and palette."""

    option_id: str
    option_identity_sha256: str
    contract_identity_sha256: str
    design_rationale: str
    claimed_insight_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _FINITE_OPTION_ID.fullmatch(
            self.option_id
        ) is None:
            raise ValueError(
                "option_id must use the closed lowercase identifier grammar"
            )
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(self.contract_identity_sha256, "contract_identity_sha256")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must be an exact tuple of non-empty strings"
            )


def resolve_finite_variation_selection(
    contract: FiniteVariationContract,
    draft: FiniteVariationSelectionDraft,
) -> FiniteVariationOption:
    """Verify selection receipts and return the one sealed full child."""

    validate_finite_variation_contract(contract)
    if type(draft) is not FiniteVariationSelectionDraft:
        raise TypeError("draft must be an exact FiniteVariationSelectionDraft")
    FiniteVariationSelectionDraft.__post_init__(draft)
    if draft.contract_identity_sha256 != contract.identity_sha256:
        raise ValueError("selection draft is bound to a different finite contract")
    option = contract.resolve(draft.option_id)
    if draft.option_identity_sha256 != option.identity_sha256:
        raise ValueError("selection draft is bound to a different finite option")
    return option


@dataclass(frozen=True, slots=True)
class InsightDraft:
    claim: str
    trigger: str
    mechanism: str
    affected_paths: Tuple[str, ...]
    evidence_summary: str
    confidence: float
    evidence_contrast_ids: Tuple[str, ...] = ()
    effect_predictions: Tuple[MetricEffectPrediction, ...] = ()
    recommended_option_families: Tuple[str, ...] = ()
    recommended_option_ids: Tuple[str, ...] = ()
    action_template: str | None = None
    falsification_condition: str | None = None

    def __post_init__(self) -> None:
        for name in ("claim", "trigger", "mechanism", "evidence_summary"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if type(self.affected_paths) is not tuple or any(
            type(value) is not str or not value.strip() for value in self.affected_paths
        ):
            raise TypeError("affected_paths must be a tuple of non-empty strings")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0 <= float(self.confidence) <= 1
        ):
            raise ValueError("confidence must be finite in [0,1]")
        _validate_contrast_ids(
            self.evidence_contrast_ids,
            name="evidence_contrast_ids",
        )
        advanced_fields_present = (
            bool(self.effect_predictions)
            or bool(self.recommended_option_families)
            or bool(self.recommended_option_ids)
            or self.action_template is not None
            or self.falsification_condition is not None
        )
        if not advanced_fields_present:
            return
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not MetricEffectPrediction
            for value in self.effect_predictions
        ):
            raise TypeError(
                "effect_predictions must contain exact MetricEffectPrediction values"
            )
        if not self.effect_predictions:
            raise ValueError("advanced insights require effect_predictions")
        for prediction in self.effect_predictions:
            MetricEffectPrediction.__post_init__(prediction)
        metric_ids = tuple(
            prediction.metric_id for prediction in self.effect_predictions
        )
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError(
                "effect_predictions must be unique and ordered by metric_id"
            )
        _validate_canonical_tokens(
            self.recommended_option_families,
            name="recommended_option_families",
            pattern=_FINITE_OPTION_FAMILY,
        )
        if self.recommended_option_ids:
            _validate_canonical_tokens(
                self.recommended_option_ids,
                name="recommended_option_ids",
                pattern=_FINITE_OPTION_ID,
            )
        for name in ("action_template", "falsification_condition"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    f"advanced insights require canonical non-empty {name}"
                )

    @property
    def has_intervention_contract(self) -> bool:
        """Whether the opt-in actionable/falsifiable fields are populated."""

        return bool(self.effect_predictions)

    def intervention_record(self) -> dict[str, object] | None:
        """Return a detached JSON-ready projection only for advanced insights."""

        InsightDraft.__post_init__(self)
        if not self.has_intervention_contract:
            return None
        assert self.action_template is not None
        assert self.falsification_condition is not None
        return {
            "effect_predictions": [
                {
                    "metric_id": prediction.metric_id,
                    "direction": prediction.direction.value,
                }
                for prediction in self.effect_predictions
            ],
            "recommended_option_families": list(
                self.recommended_option_families
            ),
            "recommended_option_ids": list(self.recommended_option_ids),
            "action_template": self.action_template,
            "falsification_condition": self.falsification_condition,
        }

    def content_record(self) -> dict[str, object]:
        """Return the complete immutable card content used by treatment binding."""

        InsightDraft.__post_init__(self)
        return {
            "schema_version": 1,
            "claim": self.claim,
            "trigger": self.trigger,
            "mechanism": self.mechanism,
            "affected_paths": list(self.affected_paths),
            "evidence_summary": self.evidence_summary,
            "confidence": float(self.confidence),
            "evidence_contrast_ids": list(self.evidence_contrast_ids),
            "effect_predictions": [
                {
                    "metric_id": prediction.metric_id,
                    "direction": prediction.direction.value,
                }
                for prediction in self.effect_predictions
            ],
            "recommended_option_families": list(
                self.recommended_option_families
            ),
            "recommended_option_ids": list(self.recommended_option_ids),
            "action_template": self.action_template,
            "falsification_condition": self.falsification_condition,
        }

    @property
    def content_sha256(self) -> str:
        """Bind prose, evidence, predictions, and exact finite actions."""

        return hashlib.sha256(
            _INSIGHT_DRAFT_CONTENT_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.content_record()))
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReflectionInsightContract:
    """Request-scoped vocabulary for actionable, falsifiable reflection.

    Metric identifiers are benchmark-owned and name quantities whose numeric
    direction can be adjudicated later. Option families are drawn from the
    finite variation vocabulary and remain stable when parent-specific option
    IDs change. Keeping this contract request-scoped avoids imposing any one
    benchmark's metrics or action taxonomy on the core workflow.
    """

    required_metric_ids: Tuple[str, ...]
    allowed_option_families: Tuple[str, ...]
    allowed_option_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_canonical_tokens(
            self.required_metric_ids,
            name="required_metric_ids",
            pattern=_REFLECTION_METRIC_ID,
        )
        _validate_canonical_tokens(
            self.allowed_option_families,
            name="allowed_option_families",
            pattern=_FINITE_OPTION_FAMILY,
        )
        if self.allowed_option_ids:
            _validate_canonical_tokens(
                self.allowed_option_ids,
                name="allowed_option_ids",
                pattern=_FINITE_OPTION_ID,
            )

    def to_record(self) -> dict[str, object]:
        ReflectionInsightContract.__post_init__(self)
        return {
            "schema_version": 2,
            "contract_identity_sha256": self.identity_sha256,
            "required_metric_ids": list(self.required_metric_ids),
            "allowed_option_families": list(self.allowed_option_families),
            "allowed_option_ids": list(self.allowed_option_ids),
            "direction_vocabulary": [
                direction.value for direction in MetricEffectDirection
            ],
        }

    @property
    def identity_sha256(self) -> str:
        """Bind exact ordered metric/action vocabularies for replay."""

        ReflectionInsightContract.__post_init__(self)
        payload = freeze_json(
            {
                "schema_version": 2,
                "required_metric_ids": list(self.required_metric_ids),
                "allowed_option_families": list(self.allowed_option_families),
                "allowed_option_ids": list(self.allowed_option_ids),
                "direction_vocabulary": [
                    direction.value for direction in MetricEffectDirection
                ],
            }
        )
        return hashlib.sha256(
            _REFLECTION_INSIGHT_CONTRACT_DOMAIN
            + canonical_typed_json_bytes(payload)
        ).hexdigest()


def validate_reflection_insight_draft(
    draft: InsightDraft,
    contract: ReflectionInsightContract,
    *,
    allow_all_unknown: bool = False,
    allow_missing_evidence: bool = False,
) -> None:
    """Verify exact metric coverage and finite-vocabulary actionability."""

    if type(allow_all_unknown) is not bool:
        raise TypeError("allow_all_unknown must be an exact boolean")
    if type(allow_missing_evidence) is not bool:
        raise TypeError("allow_missing_evidence must be an exact boolean")
    if type(draft) is not InsightDraft:
        raise TypeError("draft must be an exact InsightDraft")
    InsightDraft.__post_init__(draft)
    if type(contract) is not ReflectionInsightContract:
        raise TypeError("contract must be an exact ReflectionInsightContract")
    ReflectionInsightContract.__post_init__(contract)
    if not draft.has_intervention_contract:
        raise ValueError("insight is missing the advanced intervention contract")
    if not allow_missing_evidence and not draft.evidence_contrast_ids:
        raise ValueError(
            "an outcome-grounded intervention must cite at least one evidence contrast"
        )
    predicted_metric_ids = tuple(
        prediction.metric_id for prediction in draft.effect_predictions
    )
    if predicted_metric_ids != contract.required_metric_ids:
        raise ValueError(
            "effect predictions must cover the exact required metric identifiers"
        )
    if not set(draft.recommended_option_families).issubset(
        contract.allowed_option_families
    ):
        raise ValueError(
            "recommended option families escape the finite action vocabulary"
        )
    if contract.allowed_option_ids:
        if not draft.recommended_option_ids:
            raise ValueError(
                "an exact-action insight must recommend at least one option ID"
            )
        if not set(draft.recommended_option_ids).issubset(
            contract.allowed_option_ids
        ):
            raise ValueError(
                "recommended option IDs escape the finite action vocabulary"
            )
    elif draft.recommended_option_ids:
        raise ValueError(
            "recommended option IDs require a request-scoped exact-action vocabulary"
        )
    if not allow_all_unknown and all(
        prediction.direction is MetricEffectDirection.UNKNOWN
        for prediction in draft.effect_predictions
    ):
        raise ValueError(
            "an outcome-grounded intervention must make a directional prediction"
        )


@dataclass(frozen=True, slots=True)
class AgenticCallTelemetry:
    requested_model: str
    resolved_model: str
    resolved_provider: str
    provider_response_id: str | None
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: Decimal | None
    latency_ns: int
    attempt_count: int = 1

    def __post_init__(self) -> None:
        for name in ("requested_model", "resolved_model", "resolved_provider"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "latency_ns",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.attempt_count) is not int or self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")


@dataclass(frozen=True, slots=True)
class VariationGenerationRequest:
    call_id: LLMCallId
    operation: str
    prompt: str
    candidate_model: type
    max_output_tokens: int = 2_048
    temperature: float | None = None
    atomic_mutation_contract: AtomicMutationOutputContract | None = None
    finite_variation_contract: FiniteVariationContract | None = None

    def __post_init__(self) -> None:
        if (
            self.atomic_mutation_contract is not None
            and self.finite_variation_contract is not None
        ):
            raise ValueError(
                "atomic and finite variation contracts are mutually exclusive"
            )
        if self.atomic_mutation_contract is not None:
            if type(self.atomic_mutation_contract) is not AtomicMutationOutputContract:
                raise TypeError(
                    "atomic_mutation_contract must be an exact "
                    "AtomicMutationOutputContract"
                )
            AtomicMutationOutputContract.__post_init__(
                self.atomic_mutation_contract
            )
        if self.finite_variation_contract is not None:
            validate_finite_variation_contract(self.finite_variation_contract)


@dataclass(frozen=True, slots=True)
class ReflectionGenerationRequest:
    call_id: LLMCallId
    operation: str
    prompt: str
    max_insights: int = 4
    min_insights: int = 0
    max_output_tokens: int = 2_048
    temperature: float | None = None
    available_contrast_ids: Tuple[str, ...] = ()
    insight_contract: ReflectionInsightContract | None = None

    def __post_init__(self) -> None:
        if type(self.max_insights) is not int or not 1 <= self.max_insights <= 16:
            raise ValueError("max_insights must lie in [1,16]")
        if (
            type(self.min_insights) is not int
            or not 0 <= self.min_insights <= self.max_insights
        ):
            raise ValueError("min_insights must lie in [0,max_insights]")
        _validate_contrast_ids(
            self.available_contrast_ids,
            name="available_contrast_ids",
        )
        if self.insight_contract is not None:
            if type(self.insight_contract) is not ReflectionInsightContract:
                raise TypeError(
                    "insight_contract must be an exact ReflectionInsightContract"
                )
            ReflectionInsightContract.__post_init__(self.insight_contract)


@dataclass(frozen=True, slots=True)
class VariationGenerationResult:
    draft: CandidateDraft | AtomicMutationDraft | FiniteVariationSelectionDraft
    telemetry: AgenticCallTelemetry


@dataclass(frozen=True, slots=True)
class ReflectionGenerationResult:
    insights: Tuple[InsightDraft, ...]
    telemetry: AgenticCallTelemetry


@runtime_checkable
class AgenticGenerator(Protocol):
    async def propose(
        self, request: VariationGenerationRequest
    ) -> VariationGenerationResult: ...

    async def reflect(
        self, request: ReflectionGenerationRequest
    ) -> ReflectionGenerationResult: ...


__all__ = [
    "AgenticCallTelemetry",
    "AgenticGenerator",
    "AtomicMutationDraft",
    "AtomicMutationOutputContract",
    "CandidateDraft",
    "ConflictResolutionDraft",
    "FiniteVariationSelectionDraft",
    "InsightDraft",
    "MetricEffectDirection",
    "MetricEffectPrediction",
    "ReflectionGenerationRequest",
    "ReflectionGenerationResult",
    "ReflectionInsightContract",
    "resolve_finite_variation_selection",
    "SourceAttribution",
    "VariationGenerationRequest",
    "VariationGenerationResult",
    "validate_reflection_insight_draft",
]
