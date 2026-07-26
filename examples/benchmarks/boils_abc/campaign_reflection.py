"""BOiLS semantic vocabulary for generic identifiable campaign reflection.

BOiLS owns only its metric, decision-path, and finite-action vocabulary.  The
generic application layer owns evidence projection, provider prompts, citation
resolution, occurrence lineage, finite-action bindings, and the canonical
campaign-learning envelope.  Consequently the production path consumes only
``CampaignIdentifiableReflectionInput`` direct single-mutation evidence.

The recombination-contrast API retained at the bottom of this module is a
deprecated compatibility surface for historical artifacts.  It must not be
used by new campaign composition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import warnings
from typing import Any

from agent_evolve.agentic import (
    MetricComparisonAnchorKind,
    OptimizationSemantics,
    ReflectionConsumerScope,
    ReflectionEvidenceCatalog,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    ReflectionInsightKind,
)
from agent_evolve.application.identifiable_reflection_learning import (
    build_identifiable_campaign_reflection_learning_envelope,
)
from agent_evolve.application.identifiable_reflection_request import (
    bind_reflection_contract_to_evidence_actions,
    build_identifiable_reflection_generation_request,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import ReflectionGenerationResult
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection

from .actions import SEQUENCE_LENGTH
from .detailed_evaluation import TOTAL_LEVELS, TOTAL_LUT_COUNT
from .variation_catalog import ACTION_FAMILIES


OBJECTIVE_IDS = tuple(sorted((TOTAL_LEVELS, TOTAL_LUT_COUNT)))
REFLECTION_DECISION_PATHS = tuple(
    sorted(f"$.sequence[{index}]" for index in range(SEQUENCE_LENGTH))
)
REFLECTION_OPTION_FAMILIES = tuple(sorted(set(ACTION_FAMILIES.values())))
REFLECTION_INSIGHT_KINDS = (
    ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
)
REFLECTION_CONSUMER_SCOPES = (ReflectionConsumerScope.MUTATION_SELECTION,)
REFLECTION_COMPARISON_ANCHORS = (MetricComparisonAnchorKind.CURRENT_PARENT,)
LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_ID = "boils_abc_recombination_contrast"
LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_VERSION = 1
LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-abc-recombination-contrast-facts:v1"
).hexdigest()
# Explicit-import compatibility for the stale generic-campaign runner.  These
# aliases are intentionally absent from ``__all__`` and should be deleted when
# that runner moves to ``CampaignIdentifiableReflectionInput``.
BOILS_REFLECTION_FACT_SCHEMA_ID = LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_ID
BOILS_REFLECTION_FACT_SCHEMA_VERSION = (
    LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_VERSION
)
BOILS_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256 = (
    LEGACY_BOILS_RECOMBINATION_FACT_SCHEMA_DEFINITION_SHA256
)
_LEGACY_RECOMBINATION_WARNING = (
    "BOiLS recombination-derived reflection is deprecated; use sealed "
    "CampaignIdentifiableReflectionInput with the generic identifiable "
    "request and campaign-learning builders"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("BOiLS reflection record did not freeze to an object")
    return result


def boils_reflection_contract(
    allowed_option_families: tuple[str, ...] = REFLECTION_OPTION_FAMILIES,
) -> ReflectionInsightContract:
    """Bind reflections to exact BOiLS metrics, paths, and action families."""

    if (
        type(allowed_option_families) is not tuple
        or not allowed_option_families
        or allowed_option_families
        != tuple(sorted(set(allowed_option_families)))
        or not set(allowed_option_families).issubset(REFLECTION_OPTION_FAMILIES)
    ):
        raise ValueError("allowed_option_families must be a canonical BOiLS subset")
    return ReflectionInsightContract(
        required_metric_ids=OBJECTIVE_IDS,
        allowed_option_families=allowed_option_families,
        allowed_decision_paths=REFLECTION_DECISION_PATHS,
        allowed_insight_kinds=REFLECTION_INSIGHT_KINDS,
        allowed_consumer_scopes=REFLECTION_CONSUMER_SCOPES,
        allowed_comparison_anchor_kinds=REFLECTION_COMPARISON_ANCHORS,
        allowed_factor_capabilities=allowed_option_families,
    )


def _validate_boils_optimization_semantics(
    optimization_semantics: OptimizationSemantics,
) -> None:
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    OptimizationSemantics.__post_init__(optimization_semantics)
    decision_metrics = DecisionMetricProjection.from_optimization_semantics(
        optimization_semantics
    )
    if decision_metrics.metric_ids != OBJECTIVE_IDS:
        raise ValueError(
            "optimization semantics decision metrics differ from BOiLS objectives"
        )


def build_boils_identifiable_reflection_request(
    *,
    call_id: LLMCallId,
    reflection_input: CampaignIdentifiableReflectionInput,
    optimization_semantics: OptimizationSemantics,
    max_output_tokens: int,
    temperature: float | None,
    allowed_option_families: tuple[str, ...] = REFLECTION_OPTION_FAMILIES,
    min_insights: int = 1,
    max_insights: int = 2,
) -> ReflectionGenerationRequest:
    """Bind BOiLS semantics to the generic identifiable request constructor."""

    if type(reflection_input) is not CampaignIdentifiableReflectionInput:
        raise TypeError("reflection_input must be exact")
    CampaignIdentifiableReflectionInput.__post_init__(reflection_input)
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive exact integer")
    if temperature is not None and (
        type(temperature) is not float or not math.isfinite(temperature)
    ):
        raise ValueError("temperature must be a finite exact float or None")
    _validate_boils_optimization_semantics(optimization_semantics)
    exact_contract = bind_reflection_contract_to_evidence_actions(
        boils_reflection_contract(allowed_option_families),
        reflection_input.evidence,
    )
    return build_identifiable_reflection_generation_request(
        call_id=call_id,
        evidence=reflection_input.evidence,
        insight_contract=exact_contract,
        optimization_semantics=optimization_semantics,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        min_insights=min_insights,
        max_insights=max_insights,
    )


def build_boils_identifiable_reflection_learning_envelope(
    *,
    reflection_input: CampaignIdentifiableReflectionInput,
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
    optimization_semantics: OptimizationSemantics,
) -> FrozenJsonObject:
    """Delegate BOiLS learning lineage to the generic canonical projection."""

    _validate_boils_optimization_semantics(optimization_semantics)
    if type(request) is not ReflectionGenerationRequest:
        raise TypeError("request must be exact")
    ReflectionGenerationRequest.__post_init__(request)
    contract = request.insight_contract
    if contract is None:
        raise ValueError("identifiable BOiLS request lost its semantic contract")
    expected = bind_reflection_contract_to_evidence_actions(
        boils_reflection_contract(contract.allowed_option_families),
        reflection_input.evidence,
    )
    if contract != expected:
        raise ValueError("reflection request carries a foreign BOiLS contract")
    return build_identifiable_campaign_reflection_learning_envelope(
        reflection_input=reflection_input,
        request=request,
        result=result,
        optimization_semantics=optimization_semantics,
    )


@dataclass(frozen=True, slots=True)
class BoilsReflectionContrast:
    """Deprecated historical recombination outcome projection.

    New campaigns must use ``IdentifiableMutationReflectionContrast`` from the
    authenticated campaign evidence registry.
    """

    contrast_id: str
    wave_ordinal: int
    selection_role: str
    source_option_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    source_parent_objectives: tuple[FrozenJsonObject, ...]
    target_objectives: FrozenJsonObject
    reward_hex: str
    dominates_any_parent: bool
    better_than_any_parent: bool

    def __post_init__(self) -> None:
        if type(self.contrast_id) is not str or len(self.contrast_id) != 64:
            raise ValueError("contrast_id must be a lowercase SHA-256 identity")
        try:
            bytes.fromhex(self.contrast_id)
        except ValueError as error:
            raise ValueError(
                "contrast_id must be a lowercase SHA-256 identity"
            ) from error
        if self.contrast_id != self.contrast_id.lower():
            raise ValueError("contrast_id must be a lowercase SHA-256 identity")
        if type(self.wave_ordinal) is not int or self.wave_ordinal <= 0:
            raise ValueError("wave_ordinal must be a positive exact integer")
        if type(self.selection_role) is not str or not self.selection_role:
            raise ValueError("selection_role must be a non-empty string")
        for name, values in (
            ("source_option_ids", self.source_option_ids),
            ("source_families", self.source_families),
        ):
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise TypeError(f"{name} must contain non-empty strings")
        if not set(self.source_families).issubset(REFLECTION_OPTION_FAMILIES):
            raise ValueError("source_families escaped the BOiLS vocabulary")
        if (
            type(self.source_parent_objectives) is not tuple
            or not self.source_parent_objectives
            or any(
                type(value) is not FrozenJsonObject
                for value in self.source_parent_objectives
            )
        ):
            raise TypeError("source_parent_objectives must be frozen objects")
        if type(self.target_objectives) is not FrozenJsonObject:
            raise TypeError("target_objectives must be a frozen object")
        for record in (*self.source_parent_objectives, self.target_objectives):
            if tuple(sorted(thaw_json(record))) != OBJECTIVE_IDS:
                raise ValueError("reflection objectives differ from BOiLS metrics")
        if type(self.reward_hex) is not str:
            raise TypeError("reward_hex must be a string")
        try:
            reward = float.fromhex(self.reward_hex)
        except ValueError as error:
            raise ValueError("reward_hex must encode binary64") from error
        if not math.isfinite(reward):
            raise ValueError("reward_hex must encode finite binary64")
        if type(self.dominates_any_parent) is not bool:
            raise TypeError("dominates_any_parent must be an exact bool")
        if type(self.better_than_any_parent) is not bool:
            raise TypeError("better_than_any_parent must be an exact bool")

    def to_prompt_record(self, *, evidence_citation_key: str) -> dict[str, object]:
        self.__post_init__()
        if type(evidence_citation_key) is not str or not evidence_citation_key:
            raise ValueError("evidence_citation_key must be non-empty")
        return {
            "contrast_id": self.contrast_id,
            "evidence_citation_key": evidence_citation_key,
            "wave_ordinal": self.wave_ordinal,
            "selection_role": self.selection_role,
            "source_option_ids": list(self.source_option_ids),
            "source_families": list(self.source_families),
            "source_parent_objectives": [
                thaw_json(value) for value in self.source_parent_objectives
            ],
            "target_objectives": thaw_json(self.target_objectives),
            "reward_hex": self.reward_hex,
            "dominates_any_parent": self.dominates_any_parent,
            "better_than_any_parent": self.better_than_any_parent,
        }


def normalize_boils_reflection_contrasts(
    source_results: tuple[Any, ...],
) -> tuple[BoilsReflectionContrast, ...]:
    """Deprecated: project historical recombination results."""

    warnings.warn(
        _LEGACY_RECOMBINATION_WARNING,
        DeprecationWarning,
        stacklevel=2,
    )

    if type(source_results) is not tuple:
        raise TypeError("source_results must be an exact tuple")
    contrasts: list[BoilsReflectionContrast] = []
    for wave_ordinal, result in enumerate(source_results, start=1):
        receipt = result.receipt
        for member, outcome in zip(receipt.members, result.outcomes, strict=True):
            candidate = outcome.candidate
            if candidate is None:
                raise ValueError("reflection source lacks an evaluated candidate")
            contrasts.append(
                BoilsReflectionContrast(
                    contrast_id=member.outcome_sha256,
                    wave_ordinal=wave_ordinal,
                    selection_role=member.selection_role,
                    source_option_ids=tuple(member.source_option_ids),
                    source_families=tuple(member.source_families),
                    source_parent_objectives=tuple(
                        _object(dict(parent.objective_map))
                        for parent in outcome.prepared.plan.parents
                    ),
                    target_objectives=_object(dict(candidate.objective_map)),
                    reward_hex=outcome.reward.hex(),
                    dominates_any_parent=outcome.dominates_any_parent,
                    better_than_any_parent=outcome.better_than_any_parent,
                )
            )
    if not contrasts:
        raise ValueError("reflection requires at least one evaluated contrast")
    return tuple(contrasts)


def build_boils_reflection_generation_request(
    *,
    call_id: LLMCallId,
    contrasts: tuple[BoilsReflectionContrast, ...],
    allowed_option_families: tuple[str, ...],
    max_output_tokens: int,
    temperature: float | None,
) -> ReflectionGenerationRequest:
    """Deprecated: build a request from non-identifiable recombinations."""

    warnings.warn(
        _LEGACY_RECOMBINATION_WARNING,
        DeprecationWarning,
        stacklevel=2,
    )

    if type(call_id) is not LLMCallId:
        raise TypeError("call_id must be an exact LLMCallId")
    if (
        type(contrasts) is not tuple
        or not contrasts
        or any(type(value) is not BoilsReflectionContrast for value in contrasts)
    ):
        raise TypeError("contrasts must contain exact BoilsReflectionContrast values")
    for contrast in contrasts:
        BoilsReflectionContrast.__post_init__(contrast)
    contract = boils_reflection_contract(allowed_option_families)
    available = tuple(sorted(contrast.contrast_id for contrast in contrasts))
    if len(set(available)) != len(available):
        raise ValueError("contrast identities must be unique")
    catalog = ReflectionEvidenceCatalog.from_contrast_ids(available)
    prompt_contrasts = [
        contrast.to_prompt_record(
            evidence_citation_key=catalog.citation_key_for_contrast_id(
                contrast.contrast_id
            )
        )
        for contrast in contrasts
    ]
    prompt = json.dumps(
        {
            "task": (
                "Derive one or two falsifiable BOiLS hypotheses from the exact "
                "recombination contrasts for prospective use by a later finite "
                "typed-mutation selector. Each hypothesis must name exactly one "
                "allowed sequence path, predict both required minimization metrics "
                "relative to the current parent, recommend only allowed action "
                "families, and cite request-scoped evidence keys. Full contrast "
                "identities are authenticated context and must not be copied. The "
                "outputs remain quarantined unverified hypotheses."
            ),
            "objectives": list(OBJECTIVE_IDS),
            "contrasts": prompt_contrasts,
            "quarantine": "until a later preregistered diagnostic block closes",
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ReflectionGenerationRequest(
        call_id=call_id,
        operation="extract_insights",
        prompt=prompt,
        max_insights=2,
        min_insights=1,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        available_contrast_ids=available,
        insight_contract=contract,
        evidence_catalog=catalog,
    )


def boils_reflection_request_construction_record(
    request: ReflectionGenerationRequest,
) -> dict[str, object]:
    """Deprecated: seal a historical recombination-reflection request."""

    warnings.warn(
        _LEGACY_RECOMBINATION_WARNING,
        DeprecationWarning,
        stacklevel=2,
    )

    ReflectionGenerationRequest.__post_init__(request)
    catalog = request.evidence_catalog
    contract = request.insight_contract
    if catalog is None or contract is None:
        raise ValueError("BOiLS reflection requires catalog and semantic contract")
    prompt = json.loads(request.prompt)
    prompt_contrasts = prompt.get("contrasts") if type(prompt) is dict else None
    if type(prompt_contrasts) is not list or any(
        type(value) is not dict for value in prompt_contrasts
    ):
        raise ValueError("reflection prompt contrasts must be an object list")
    expected = tuple(
        sorted((entry.contrast_id, entry.citation_key) for entry in catalog.entries)
    )
    observed = tuple(
        sorted(
            (
                str(value.get("contrast_id")),
                str(value.get("evidence_citation_key")),
            )
            for value in prompt_contrasts
        )
    )
    mapping = {
        "schema_version": 1,
        "entries": [
            {"contrast_id": contrast_id, "evidence_citation_key": key}
            for contrast_id, key in observed
        ],
    }
    identity = {
        "schema_version": 1,
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_sha256": _sha(request.prompt),
        "max_insights": request.max_insights,
        "min_insights": request.min_insights,
        "max_output_tokens": request.max_output_tokens,
        "temperature_hex": (
            None if request.temperature is None else request.temperature.hex()
        ),
        "available_contrast_ids": list(request.available_contrast_ids),
        "evidence_catalog_identity_sha256": catalog.catalog_identity_sha256,
        "insight_contract_identity_sha256": contract.identity_sha256,
        "evidence_citation_mapping_sha256": typed_json_sha256(_object(mapping)),
    }
    return {
        **identity,
        "request_identity_sha256": typed_json_sha256(_object(identity)),
        "prompt_utf8_bytes": len(request.prompt.encode("utf-8", errors="strict")),
        "evidence_citation_mapping": mapping["entries"],
        "exact_evidence_citation_mapping": observed == expected,
        "no_legacy_evidence_key": all(
            "evidence_key" not in value for value in prompt_contrasts
        ),
    }


__all__ = [
    "OBJECTIVE_IDS",
    "REFLECTION_DECISION_PATHS",
    "REFLECTION_OPTION_FAMILIES",
    "boils_reflection_contract",
    "build_boils_identifiable_reflection_learning_envelope",
    "build_boils_identifiable_reflection_request",
]
