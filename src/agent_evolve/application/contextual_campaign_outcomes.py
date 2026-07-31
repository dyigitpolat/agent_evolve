"""Authenticated portfolio outcomes projected into contextual search credit.

The projection consumes only framework-level source labels, finite-action
families, exact option identities, dispositions, and normalized marginal
utilities supplied by the campaign's archive-utility port.  It does not inspect
workload fields, objective names, model prose, or provider metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.contextual_search_controller import (
    ContextualSearchObservation,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioMemberDisposition,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.variation_source import finite_variation_operator_id


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_BATCH_DOMAIN = b"agent-evolve:contextual-portfolio-outcome-batch:v1\x00"
CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_ID = (
    "normalized_fixed_reference_contextual_portfolio_outcomes"
)
CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_VERSION = 2
CONTEXTUAL_PORTFOLIO_OUTCOME_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:normalized-fixed-reference-contextual-portfolio-outcomes:v2;"
    b"source-labels=framework-injected;"
    b"operator=sealed-finite-option-evaluation-operator;"
    b"utility=nonnegative-campaign-archive-marginal;normalization=stage-sum;"
    b"infeasible-yield=zero;final-persistence=delayed;descendant=delayed;"
    b"workload-objective-model-provider-fields=false"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class ContextualPortfolioOutcomeBatch:
    campaign_scope_sha256: str
    wave_index: int
    wave_request_sha256s: tuple[str, ...]
    result_receipt_sha256s: tuple[str, ...]
    observations: tuple[ContextualSearchObservation, ...]
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.campaign_scope_sha256, "campaign_scope_sha256")
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be positive")
        for name in ("wave_request_sha256s", "result_receipt_sha256s"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty exact tuple")
            for value in values:
                require_sha256(value, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if type(self.observations) is not tuple or not self.observations:
            raise ValueError("observations must be a non-empty exact tuple")
        for value in self.observations:
            if type(value) is not ContextualSearchObservation:
                raise TypeError("observations must contain exact controller values")
            value.__post_init__()
            if (
                value.campaign_scope_sha256 != self.campaign_scope_sha256
                or value.wave_index != self.wave_index
            ):
                raise ValueError("contextual observation differs from its batch")
        if tuple(value.observation_sha256 for value in self.observations) != tuple(
            sorted({value.observation_sha256 for value in self.observations})
        ):
            raise ValueError("observations must use canonical unique identities")
        object.__setattr__(
            self,
            "batch_sha256",
            hashlib.sha256(
                _BATCH_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "wave_index": self.wave_index,
            "wave_request_sha256s": list(self.wave_request_sha256s),
            "result_receipt_sha256s": list(self.result_receipt_sha256s),
            "observation_sha256s": [
                value.observation_sha256 for value in self.observations
            ],
            "policy": {
                "policy_id": CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_ID,
                "policy_version": CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_VERSION,
                "definition_sha256": (
                    CONTEXTUAL_PORTFOLIO_OUTCOME_DEFINITION_SHA256
                ),
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "observations": [value.to_record() for value in self.observations],
            "batch_sha256": self.batch_sha256,
        }


def observe_contextual_portfolio_outcomes(
    *,
    campaign_scope_sha256: str,
    wave_index: int,
    waves: tuple[PortfolioVariationWaveRequest, ...],
    results: tuple[PortfolioVariationWaveResult, ...],
    selected_source_ids: tuple[tuple[str, ...], ...],
    marginal_utilities: tuple[tuple[float, ...], ...],
) -> ContextualPortfolioOutcomeBatch:
    """Join selected sources and fixed-reference utility to evaluated actions."""

    require_sha256(campaign_scope_sha256, "campaign_scope_sha256")
    if type(wave_index) is not int or wave_index <= 0:
        raise ValueError("wave_index must be positive")
    if (
        type(waves) is not tuple
        or type(results) is not tuple
        or type(selected_source_ids) is not tuple
        or type(marginal_utilities) is not tuple
        or not waves
        or not (
            len(waves)
            == len(results)
            == len(selected_source_ids)
            == len(marginal_utilities)
        )
    ):
        raise ValueError("contextual outcome inputs must be equal non-empty tuples")
    flattened_utilities: list[float] = []
    for wave, result, sources, utilities in zip(
        waves,
        results,
        selected_source_ids,
        marginal_utilities,
        strict=True,
    ):
        if type(wave) is not PortfolioVariationWaveRequest:
            raise TypeError("waves must contain exact portfolio requests")
        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("results must contain exact portfolio results")
        PortfolioVariationWaveRequest.__post_init__(wave)
        PortfolioVariationWaveResult.__post_init__(result)
        members = result.receipt.members
        if (
            type(sources) is not tuple
            or type(utilities) is not tuple
            or len(sources) != len(members)
            or len(utilities) != len(members)
        ):
            raise ValueError("source/utility rows must exactly cover evaluated members")
        for source in sources:
            if type(source) is not str or _TOKEN.fullmatch(source) is None:
                raise ValueError("source ID must use the closed token grammar")
        for member, utility in zip(members, utilities, strict=True):
            if (
                type(utility) is not float
                or not math.isfinite(utility)
                or not 0.0 <= utility <= 1.0
            ):
                raise ValueError("normalized marginal utility must lie in [0, 1]")
            if (
                member.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
                and utility != 0.0
            ):
                raise ValueError("infeasible candidates cannot carry marginal utility")
            flattened_utilities.append(utility)
    total_utility = sum(flattened_utilities)
    observations: list[ContextualSearchObservation] = []
    for wave, result, sources, utilities in zip(
        waves,
        results,
        selected_source_ids,
        marginal_utilities,
        strict=True,
    ):
        if not result.action_attributions:
            raise ValueError("contextual credit requires exact action attributions")
        option_by_id = {
            value.option_id: value
            for value in wave.selection_request.finite_variation_contract.options
        }
        for member, attribution, source, utility in zip(
            result.receipt.members,
            result.action_attributions,
            sources,
            utilities,
            strict=True,
        ):
            option = option_by_id.get(member.materialization.option_id)
            if option is None:
                raise ValueError("evaluated action escapes its sealed finite contract")
            if option.identity_sha256 != member.materialization.option_identity_sha256:
                raise ValueError("evaluated action identity differs from its contract")
            observations.append(
                ContextualSearchObservation(
                    campaign_scope_sha256=campaign_scope_sha256,
                    wave_index=wave_index,
                    source_id=source,
                    operator_id=finite_variation_operator_id(option),
                    option_identity_sha256=(
                        member.materialization.option_identity_sha256
                    ),
                    parent_context_sha256=(
                        wave.selection_request.context_sha256
                    ),
                    feasible=(
                        member.disposition is PortfolioMemberDisposition.SCORED
                    ),
                    positive_marginal_utility=utility > 0.0,
                    normalized_marginal_utility=utility,
                    marginal_utility_share=(
                        0.0 if total_utility == 0.0 else utility / total_utility
                    ),
                    final_front_persisted=None,
                    useful_descendant_observed=None,
                    source_distance=0.0,
                    candidate_id=member.materialization.candidate_id,
                )
            )
    return ContextualPortfolioOutcomeBatch(
        campaign_scope_sha256=campaign_scope_sha256,
        wave_index=wave_index,
        wave_request_sha256s=tuple(
            sorted(value.selection_request.request_sha256 for value in waves)
        ),
        result_receipt_sha256s=tuple(
            sorted(value.receipt.receipt_sha256 for value in results)
        ),
        observations=tuple(
            sorted(observations, key=lambda value: value.observation_sha256)
        ),
    )


__all__ = [
    "CONTEXTUAL_PORTFOLIO_OUTCOME_DEFINITION_SHA256",
    "CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_ID",
    "CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_VERSION",
    "ContextualPortfolioOutcomeBatch",
    "observe_contextual_portfolio_outcomes",
]
