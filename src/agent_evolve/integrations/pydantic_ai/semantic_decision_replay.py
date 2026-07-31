"""Sealed replay of validated hierarchical residual proposal decisions.

Provider-wire replay and semantic-decision replay have different authorities.
An output accepted after a bounded schema-repair prompt must not be presented
as an output from the original wire prompt.  Its validated domain decision can
nevertheless be replayed for the exact semantic request that produced it.
This module implements that higher boundary without changing the live policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Callable, cast

from agent_evolve.application.residual_reachability import (
    HierarchicalResidualPlan,
    ResidualProposalRole,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    HierarchicalResidualMetricForecast,
    HierarchicalResidualProposalRequest,
    HierarchicalResidualProposalSelection,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_DECISION_DOMAIN = b"agent-evolve:hierarchical-residual-proposal-decision:v2\x00"
_REPLAY_DECISION_DOMAIN = (
    b"agent-evolve:sealed-hierarchical-residual-proposal-replay:v1\x00"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def hierarchical_residual_proposal_selection_from_record(
    record: object,
    *,
    telemetry: AgenticCallTelemetry,
) -> HierarchicalResidualProposalSelection:
    """Rebuild and authenticate one durable semantic proposal decision."""

    if type(record) is not dict:
        raise TypeError("selection record must be an exact dict")
    if type(telemetry) is not AgenticCallTelemetry:
        raise TypeError("telemetry must be an exact AgenticCallTelemetry")
    AgenticCallTelemetry.__post_init__(telemetry)
    source = cast(dict[str, object], record)
    plan_records = source.get("plans")
    rationale_records = source.get("rationales")
    probability_records = source.get("probability_valid_hex")
    prediction_groups = source.get("effect_predictions")
    if type(plan_records) is not list or not plan_records:
        raise ValueError("selection record plans must be a non-empty exact list")
    if type(rationale_records) is not list:
        raise TypeError("selection record rationales must be an exact list")
    if type(probability_records) is not list:
        raise TypeError(
            "selection record probability_valid_hex must be an exact list"
        )
    if type(prediction_groups) is not list:
        raise TypeError(
            "selection record effect_predictions must be an exact list"
        )
    if not (
        len(plan_records)
        == len(rationale_records)
        == len(probability_records)
        == len(prediction_groups)
    ):
        raise ValueError("selection record member fields do not align")

    decision_sha256 = source.get("decision_sha256")
    request_sha256 = source.get("request_sha256")
    require_sha256(decision_sha256, "decision_sha256")
    require_sha256(request_sha256, "request_sha256")
    plans: list[HierarchicalResidualPlan] = []
    forecasts: list[tuple[HierarchicalResidualMetricForecast, ...]] = []
    probability_valid: list[float] = []
    rationales: list[str] = []
    draft_members: list[dict[str, object]] = []
    for plan_record, rationale, probability_hex, prediction_records in zip(
        plan_records,
        rationale_records,
        probability_records,
        prediction_groups,
        strict=True,
    ):
        if type(plan_record) is not dict:
            raise TypeError("selection plan record must be an exact dict")
        if type(rationale) is not str or not rationale:
            raise ValueError("selection rationale must be a non-empty exact str")
        if type(probability_hex) is not str:
            raise TypeError("probability_valid_hex must contain exact strings")
        if type(prediction_records) is not list or not prediction_records:
            raise ValueError(
                "effect prediction group must be a non-empty exact list"
            )
        plan_source = cast(dict[str, object], plan_record)
        component_ids = plan_source.get("component_option_ids")
        if type(component_ids) is not list:
            raise TypeError("component_option_ids must be an exact list")
        plan = HierarchicalResidualPlan(
            parent_candidate_id=CandidateId(
                cast(str, plan_source.get("parent_candidate_id"))
            ),
            parent_contract_sha256=cast(
                str, plan_source.get("parent_contract_sha256")
            ),
            action_schema_sha256=cast(
                str, plan_source.get("action_schema_sha256")
            ),
            component_option_ids=tuple(component_ids),
            role=ResidualProposalRole(cast(str, plan_source.get("role"))),
            expert_id=cast(str, plan_source.get("expert_id")),
            expert_definition_sha256=cast(
                str, plan_source.get("expert_definition_sha256")
            ),
            native_rank=cast(int, plan_source.get("native_rank")),
            decision_receipt_sha256=cast(
                str, plan_source.get("decision_receipt_sha256")
            ),
        )
        if plan.decision_receipt_sha256 != decision_sha256:
            raise ValueError("plan decision receipt differs from selection")
        parsed_predictions: list[HierarchicalResidualMetricForecast] = []
        draft_predictions: list[dict[str, object]] = []
        for prediction_record in prediction_records:
            if type(prediction_record) is not dict:
                raise TypeError("metric forecast record must be an exact dict")
            prediction_source = cast(dict[str, object], prediction_record)
            metric_id = prediction_source.get("metric_id")
            hex_values = tuple(
                prediction_source.get(name)
                for name in (
                    "p10_delta_hex",
                    "p50_delta_hex",
                    "p90_delta_hex",
                    "confidence_hex",
                )
            )
            if type(metric_id) is not str or not metric_id:
                raise ValueError("metric forecast ID must be non-empty")
            if any(type(value) is not str for value in hex_values):
                raise TypeError("metric forecast hex values must be exact strings")
            p10_hex, p50_hex, p90_hex, confidence_hex = cast(
                tuple[str, str, str, str],
                hex_values,
            )
            prediction = HierarchicalResidualMetricForecast(
                metric_id=metric_id,
                p10_delta=float.fromhex(p10_hex),
                p50_delta=float.fromhex(p50_hex),
                p90_delta=float.fromhex(p90_hex),
                confidence=float.fromhex(confidence_hex),
            )
            parsed_predictions.append(prediction)
            draft_predictions.append(
                {
                    "metric_id": prediction.metric_id,
                    "p10_delta": prediction.p10_delta,
                    "p50_delta": prediction.p50_delta,
                    "p90_delta": prediction.p90_delta,
                    "confidence": prediction.confidence,
                }
            )
        probability = float.fromhex(probability_hex)
        plans.append(plan)
        rationales.append(rationale)
        probability_valid.append(probability)
        forecasts.append(tuple(parsed_predictions))
        draft_members.append(
            {
                "parent_candidate_id": plan.parent_candidate_id.value,
                "component_option_ids": list(plan.component_option_ids),
                "role": plan.role.value,
                "probability_valid": probability,
                "effect_predictions": draft_predictions,
                "interaction_rationale": rationale,
            }
        )

    slate_rationale = source.get("slate_rationale")
    if type(slate_rationale) is not str or not slate_rationale:
        raise ValueError("slate_rationale must be a non-empty exact str")
    expected_decision_sha256 = _hash(
        _DECISION_DOMAIN,
        {
            "schema_version": 1,
            "request_sha256": request_sha256,
            "members": draft_members,
            "slate_rationale": slate_rationale,
        },
    )
    if expected_decision_sha256 != decision_sha256:
        raise ValueError("selection decision digest does not close")
    result = HierarchicalResidualProposalSelection(
        request_sha256=cast(str, request_sha256),
        decision_sha256=cast(str, decision_sha256),
        plans=tuple(plans),
        rationales=tuple(rationales),
        probability_valid=tuple(probability_valid),
        effect_predictions=tuple(forecasts),
        slate_rationale=slate_rationale,
        telemetry=telemetry,
    )
    if result.to_record() != source:
        raise ValueError("selection record is not exact canonical output")
    return result


class HierarchicalResidualProposalReplayError(RuntimeError):
    """A sealed semantic decision cannot serve the current request."""


@dataclass(slots=True)
class SealedHierarchicalResidualProposalReplayThenLivePolicy:
    """Replay exact semantic decisions, then allow an explicit live suffix."""

    source_id: str
    source_identity_sha256: str
    sealed_selections: tuple[HierarchicalResidualProposalSelection, ...]
    live_policy: Any
    allowed_live_call_ids: tuple[str, ...] = ()
    decision_receipt_sink: Callable[[dict[str, object]], None] | None = None
    _remaining: dict[str, HierarchicalResidualProposalSelection] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.source_id) is not str or _TOKEN.fullmatch(self.source_id) is None:
            raise ValueError("source_id must use the closed token grammar")
        require_sha256(self.source_identity_sha256, "source_identity_sha256")
        if (
            type(self.sealed_selections) is not tuple
            or any(
                type(value) is not HierarchicalResidualProposalSelection
                for value in self.sealed_selections
            )
        ):
            raise TypeError("sealed_selections must contain exact selections")
        for value in self.sealed_selections:
            value.__post_init__()
        request_ids = tuple(value.request_sha256 for value in self.sealed_selections)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("sealed selections repeat a semantic request")
        if not callable(getattr(self.live_policy, "select", None)):
            raise TypeError("live_policy must expose an async select method")
        if (
            type(self.allowed_live_call_ids) is not tuple
            or any(type(value) is not str for value in self.allowed_live_call_ids)
            or len(set(self.allowed_live_call_ids))
            != len(self.allowed_live_call_ids)
        ):
            raise TypeError("allowed_live_call_ids must be a unique exact tuple")
        for value in self.allowed_live_call_ids:
            LLMCallId(value).__post_init__()
        if self.decision_receipt_sink is not None and not callable(
            self.decision_receipt_sink
        ):
            raise TypeError("decision_receipt_sink must be callable or None")
        self._remaining = {
            value.request_sha256: value for value in self.sealed_selections
        }

    @property
    def remaining_selection_count(self) -> int:
        return len(self._remaining)

    def assert_consumed(self) -> None:
        if self._remaining:
            raise HierarchicalResidualProposalReplayError(
                "sealed semantic proposal decisions remain unconsumed"
            )

    def _publish(
        self,
        *,
        decision: str,
        request: HierarchicalResidualProposalRequest,
        selection: HierarchicalResidualProposalSelection,
    ) -> None:
        if self.decision_receipt_sink is None:
            return
        unsigned: dict[str, object] = {
            "schema_version": 1,
            "decision": decision,
            "source_id": self.source_id,
            "source_identity_sha256": self.source_identity_sha256,
            "current_call_id": request.call_id.value,
            "current_request_sha256": request.request_sha256,
            "selection_decision_sha256": selection.decision_sha256,
        }
        self.decision_receipt_sink(
            {
                **unsigned,
                "decision_receipt_sha256": _hash(
                    _REPLAY_DECISION_DOMAIN,
                    unsigned,
                ),
            }
        )

    async def select(
        self,
        request: HierarchicalResidualProposalRequest,
    ) -> HierarchicalResidualProposalSelection:
        request.__post_init__()
        request_sha256 = request.request_sha256
        replayed = self._remaining.pop(request_sha256, None)
        if replayed is not None:
            for plan in replayed.plans:
                if (
                    plan.action_schema_sha256
                    != request.action_schema.schema_sha256
                    or plan.expert_id != request.expert_id
                    or plan.expert_definition_sha256
                    != request.expert_definition_sha256
                    or plan.radius not in request.allowed_radii
                    or plan.role not in request.allowed_roles
                ):
                    raise HierarchicalResidualProposalReplayError(
                        "sealed semantic selection differs from current contract"
                    )
                expected_contract = request.action_schema.contract_for(
                    plan.parent_candidate_id
                )
                if (
                    expected_contract.identity_sha256
                    != plan.parent_contract_sha256
                ):
                    raise HierarchicalResidualProposalReplayError(
                        "sealed parent contract differs from current contract"
                    )
            self._publish(
                decision="replayed",
                request=request,
                selection=replayed,
            )
            return replayed
        if request.call_id.value not in self.allowed_live_call_ids:
            raise HierarchicalResidualProposalReplayError(
                "no sealed decision exists and live execution is unauthorized"
            )
        live = await self.live_policy.select(request)
        if (
            type(live) is not HierarchicalResidualProposalSelection
            or live.request_sha256 != request_sha256
        ):
            raise HierarchicalResidualProposalReplayError(
                "live policy returned a foreign semantic decision"
            )
        self._publish(
            decision="live_after_prefix",
            request=request,
            selection=live,
        )
        return live


__all__ = [
    "HierarchicalResidualProposalReplayError",
    "SealedHierarchicalResidualProposalReplayThenLivePolicy",
    "hierarchical_residual_proposal_selection_from_record",
]
