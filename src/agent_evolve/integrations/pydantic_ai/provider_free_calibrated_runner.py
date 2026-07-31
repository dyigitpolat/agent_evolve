"""Outcome-blind typed runner for calibrated-portfolio conformance assays.

The production selector exposes a complete finite action contract on its
dynamic Pydantic output type.  This runner derives one legal proposal slate
from that contract without consulting a workload adapter, objective value,
model, provider, or hidden evaluator.  It is deliberately a transport double:
its only purpose is to exercise the real prompt, schema, reconciliation, and
allocation stack before a paid or simulator-backed campaign is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, get_args

from pydantic import BaseModel

from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
    CompositionSelectionExposure,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
)
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_witness,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


class ProviderFreeCalibratedConformanceError(RuntimeError):
    """The published calibrated-output contract has no derived legal slate."""


@dataclass(frozen=True, slots=True)
class ProviderFreeCalibratedCallRecord:
    call_id: str
    proposal_option_ids: tuple[str, ...]
    supporting_card_keys_by_member: tuple[tuple[str, ...], ...]
    assigned_card_keys: tuple[str, ...]
    evaluation_witness_option_ids: tuple[str, ...]
    required_proposal_support_option_ids: tuple[str, ...]
    required_composite_proposals: int
    bounded_memory_dose: bool

    def to_record(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "proposal_option_ids": list(self.proposal_option_ids),
            "proposal_width": len(self.proposal_option_ids),
            "supporting_card_keys_by_member": [
                list(value) for value in self.supporting_card_keys_by_member
            ],
            # Preserve the established provider-free trace names while also
            # publishing the workload-neutral spelling above.
            "proposal_supporting_card_keys": [
                list(value) for value in self.supporting_card_keys_by_member
            ],
            "assigned_card_keys": list(self.assigned_card_keys),
            "evaluation_witness_option_ids": list(
                self.evaluation_witness_option_ids
            ),
            "required_proposal_support_option_ids": list(
                self.required_proposal_support_option_ids
            ),
            "required_composite_proposals": self.required_composite_proposals,
            "bounded_memory_dose": self.bounded_memory_dose,
            "provider_calls": 0,
            "outcomes_consulted": False,
        }


def _nested_model_types(annotation: object) -> tuple[type[BaseModel], ...]:
    found: list[type[BaseModel]] = []

    def visit(value: object) -> None:
        if isinstance(value, type) and issubclass(value, BaseModel):
            found.append(value)
            return
        for child in get_args(value):
            visit(child)

    visit(annotation)
    unique: list[type[BaseModel]] = []
    for value in found:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _member_model_types(output_type: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    member_field = output_type.model_fields.get("members")
    if member_field is None:
        raise ProviderFreeCalibratedConformanceError(
            "structured output omits the calibrated members field"
        )
    result = tuple(
        value
        for value in _nested_model_types(member_field.annotation)
        if hasattr(value, "required_metric_ids")
        and (
            hasattr(value, "allowed_option_ids")
            or hasattr(value, "allowed_composite_option_ids")
        )
    )
    if not result:
        raise ProviderFreeCalibratedConformanceError(
            "structured output does not publish calibrated member contracts"
        )
    return result


def _metric_ids(member_types: tuple[type[BaseModel], ...]) -> tuple[str, ...]:
    declarations = {
        tuple(getattr(value, "required_metric_ids")) for value in member_types
    }
    if len(declarations) != 1:
        raise ProviderFreeCalibratedConformanceError(
            "calibrated member variants disagree on required metrics"
        )
    result = next(iter(declarations))
    if not result or any(type(value) is not str or not value for value in result):
        raise ProviderFreeCalibratedConformanceError(
            "calibrated output publishes an invalid metric contract"
        )
    return result


def _composite_bindings(
    member_types: tuple[type[BaseModel], ...],
) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for member_type in member_types:
        local = getattr(member_type, "components_by_composite", None)
        if local is None:
            continue
        if type(local) is not dict:
            raise ProviderFreeCalibratedConformanceError(
                "composite member contract published invalid bindings"
            )
        for option_id, components in local.items():
            if (
                type(option_id) is not str
                or type(components) is not tuple
                or len(components) != 2
                or any(type(value) is not str for value in components)
            ):
                raise ProviderFreeCalibratedConformanceError(
                    "composite member contract published an invalid binding"
                )
            prior = result.get(option_id)
            if prior is not None and prior != components:
                raise ProviderFreeCalibratedConformanceError(
                    "composite member variants disagree on one binding"
                )
            result[option_id] = components
    return result


def _eligible_options(output_type: type[BaseModel]) -> tuple[Any, ...]:
    contract = getattr(output_type, "finite_variation_contract", None)
    if contract is None:
        raise ProviderFreeCalibratedConformanceError(
            "calibrated output omits its finite variation contract"
        )
    contract.__post_init__()
    option_by_id = {value.option_id: value for value in contract.options}
    ordered_pool = getattr(output_type, "ordered_common_pool_option_ids", None)
    option_ids = (
        tuple(option_by_id)
        if ordered_pool is None
        else tuple(ordered_pool)
    )
    if (
        len(option_ids) < 8
        or len(set(option_ids)) != len(option_ids)
        or any(value not in option_by_id for value in option_ids)
    ):
        raise ProviderFreeCalibratedConformanceError(
            "calibrated finite candidate universe cannot supply a unique K8"
        )
    return tuple(option_by_id[value] for value in option_ids)


def _is_composite(option: Any, bindings: dict[str, tuple[str, str]]) -> bool:
    if option.option_id in bindings:
        return True
    exposure = dict(option.metadata).get(
        COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY
    )
    if exposure == CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value:
        raise ProviderFreeCalibratedConformanceError(
            "hierarchical composite lacks its exact component binding"
        )
    return False


def _evaluation_witness(
    output_type: type[BaseModel],
    eligible_option_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if not getattr(output_type, "enforce_cross_member_constraints", True):
        return ()
    if not getattr(
        output_type,
        "require_pairwise_disjoint_parent_patches",
        False,
    ):
        return ()
    contract = output_type.finite_variation_contract
    witness = pairwise_disjoint_parent_patch_witness(
        contract,
        eligible_option_ids,
        portfolio_size=output_type.evaluation_portfolio_size,
        min_distinct_families=output_type.min_distinct_families,
        family_exposure_bounds=output_type.required_evaluation_family_bounds,
    )
    if witness is None:
        raise ProviderFreeCalibratedConformanceError(
            "finite candidate universe has no legal evaluator witness"
        )
    return witness


def _fits_hierarchy(
    option_id: str,
    *,
    chosen: set[str],
    composite_ids: set[str],
    composite_capacity: int,
    total_capacity: int,
) -> bool:
    if option_id in chosen:
        return True
    composite_count = len(chosen.intersection(composite_ids))
    atomic_count = len(chosen) - composite_count
    if option_id in composite_ids:
        composite_count += 1
    else:
        atomic_count += 1
    return (
        composite_count <= composite_capacity
        and atomic_count <= total_capacity - composite_capacity
        and composite_count + atomic_count <= total_capacity
    )


def _memory_support_choices(
    *,
    dose: BoundedPortfolioMemoryDoseContract,
    eligible: tuple[Any, ...],
    chosen: set[str],
    composite_ids: set[str],
    composite_capacity: int,
    total_capacity: int,
) -> tuple[str, ...]:
    """Choose enough compatible members without objective or prose ranking."""

    dose.__post_init__()
    eligible_by_id = {value.option_id: value for value in eligible}
    result: list[str] = []
    occupied: set[str] = set()
    for support in dose.card_supports:
        compatible = tuple(
            option_id
            for option_id, identity in support.compatible_options
            if option_id in eligible_by_id
            and eligible_by_id[option_id].identity_sha256 == identity
        )
        selected = next(
            (value for value in compatible if value in chosen and value not in occupied),
            None,
        )
        if selected is None:
            selected = next(
                (
                    value
                    for value in compatible
                    if value not in occupied
                    and _fits_hierarchy(
                        value,
                        chosen=chosen | set(result),
                        composite_ids=composite_ids,
                        composite_capacity=composite_capacity,
                        total_capacity=total_capacity,
                    )
                ),
                None,
            )
        if selected is None:
            raise ProviderFreeCalibratedConformanceError(
                f"no distinct legal support option for assigned card {support.card_key}"
            )
        result.append(selected)
        occupied.add(selected)

    lower, upper = dose.proposed_supported_member_bounds
    maximum_supported = min(
        upper,
        total_capacity - dose.minimum_unattributed_proposed_members,
    )
    if lower > maximum_supported:
        raise ProviderFreeCalibratedConformanceError(
            "memory-dose bounds cannot fit the calibrated proposal width"
        )
    compatible_union = tuple(
        option_id
        for option in eligible
        for option_id in (option.option_id,)
        if any(
            support.supports(option.option_id, option.identity_sha256)
            for support in dose.card_supports
        )
    )
    for option_id in compatible_union:
        if len(set(result)) >= lower:
            break
        if option_id in result or not _fits_hierarchy(
            option_id,
            chosen=chosen | set(result),
            composite_ids=composite_ids,
            composite_capacity=composite_capacity,
            total_capacity=total_capacity,
        ):
            continue
        result.append(option_id)
    if len(set(result)) < lower:
        raise ProviderFreeCalibratedConformanceError(
            "finite candidate universe cannot realize the memory-dose lower bound"
        )
    return tuple(dict.fromkeys(result))


def _select_option_ids(
    output_type: type[BaseModel],
    eligible: tuple[Any, ...],
    bindings: dict[str, tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    proposal_width = 8
    required_composites = getattr(output_type, "required_composite_proposals", 0)
    if type(required_composites) is not int or not (
        0 <= required_composites < proposal_width
    ):
        raise ProviderFreeCalibratedConformanceError(
            "calibrated output publishes an invalid composite count"
        )
    eligible_ids = tuple(value.option_id for value in eligible)
    composite_ids = {
        value.option_id for value in eligible if _is_composite(value, bindings)
    }
    required_support = tuple(
        sorted(getattr(output_type, "required_proposal_support_option_ids", ()))
    )
    if not set(required_support).issubset(eligible_ids):
        raise ProviderFreeCalibratedConformanceError(
            "proposal-support reservation escapes the finite universe"
        )
    witness = _evaluation_witness(output_type, eligible_ids)
    mandatory = set(required_support) | set(witness)
    if len(mandatory) > proposal_width or any(
        not _fits_hierarchy(
            option_id,
            chosen=mandatory - {option_id},
            composite_ids=composite_ids,
            composite_capacity=required_composites,
            total_capacity=proposal_width,
        )
        for option_id in mandatory
    ):
        raise ProviderFreeCalibratedConformanceError(
            "proposal reservations and evaluator witness exceed K8 capacity"
        )
    dose = getattr(output_type, "memory_dose_contract", None)
    if dose is not None:
        mandatory.update(
            _memory_support_choices(
                dose=dose,
                eligible=eligible,
                chosen=mandatory,
                composite_ids=composite_ids,
                composite_capacity=required_composites,
                total_capacity=proposal_width,
            )
        )
    for target_composite, candidate_ids in (
        (True, tuple(value for value in eligible_ids if value in composite_ids)),
        (False, tuple(value for value in eligible_ids if value not in composite_ids)),
    ):
        target_count = (
            required_composites
            if target_composite
            else proposal_width - required_composites
        )
        current_count = len(
            mandatory.intersection(composite_ids)
            if target_composite
            else mandatory - composite_ids
        )
        for option_id in candidate_ids:
            if current_count == target_count:
                break
            if option_id not in mandatory:
                mandatory.add(option_id)
                current_count += 1
        if current_count != target_count:
            raise ProviderFreeCalibratedConformanceError(
                "finite universe cannot fill the exact hierarchical K8"
            )
    selected = tuple(value for value in eligible_ids if value in mandatory)
    if (
        len(selected) != proposal_width
        or len(set(selected)) != proposal_width
        or len(set(selected).intersection(composite_ids)) != required_composites
    ):
        raise ProviderFreeCalibratedConformanceError(
            "derived calibrated proposal slate does not close"
        )
    return selected, witness


def _supporting_cards_by_option(
    *,
    output_type: type[BaseModel],
    selected: tuple[Any, ...],
) -> dict[str, tuple[str, ...]]:
    assigned_cards = tuple(sorted(output_type.assigned_card_keys))
    if not assigned_cards:
        return {option.option_id: () for option in selected}
    dose = output_type.memory_dose_contract
    if dose is None:
        return {
            option.option_id: (
                assigned_cards if index == 0 else (assigned_cards[0],)
            )
            for index, option in enumerate(selected)
        }
    dose.__post_init__()
    assignments: dict[str, list[str]] = {
        option.option_id: [] for option in selected
    }
    for support in dose.card_supports:
        target = next(
            (
                option
                for option in selected
                if len(assignments[option.option_id])
                < dose.maximum_cards_per_member
                and support.supports(option.option_id, option.identity_sha256)
            ),
            None,
        )
        if target is None:
            raise ProviderFreeCalibratedConformanceError(
                f"selected K8 cannot attribute assigned card {support.card_key}"
            )
        assignments[target.option_id].append(support.card_key)
    lower, _ = dose.proposed_supported_member_bounds
    while sum(bool(value) for value in assignments.values()) < lower:
        target_pair = next(
            (
                (option, support)
                for option in selected
                if not assignments[option.option_id]
                for support in dose.card_supports
                if support.supports(option.option_id, option.identity_sha256)
            ),
            None,
        )
        if target_pair is None:
            raise ProviderFreeCalibratedConformanceError(
                "selected K8 cannot realize its supported-member lower bound"
            )
        option, support = target_pair
        assignments[option.option_id].append(support.card_key)
    return {
        option_id: tuple(card_keys)
        for option_id, card_keys in assignments.items()
    }


class ProviderFreeCalibratedPortfolioRunner:
    """Callable local runner over the real calibrated structured-output port."""

    def __init__(self) -> None:
        self.calls = 0
        self.records: list[dict[str, object]] = []

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        request.__post_init__()
        if not issubclass(request.output_type, BaseModel):
            raise ProviderFreeCalibratedConformanceError(
                "calibrated output type must be a Pydantic model"
            )
        output_type = request.output_type
        member_types = _member_model_types(output_type)
        metric_ids = _metric_ids(member_types)
        bindings = _composite_bindings(member_types)
        eligible = _eligible_options(output_type)
        selected_ids, witness = _select_option_ids(
            output_type,
            eligible,
            bindings,
        )
        option_by_id = {value.option_id: value for value in eligible}
        selected = tuple(option_by_id[value] for value in selected_ids)
        cards_by_option = _supporting_cards_by_option(
            output_type=output_type,
            selected=selected,
        )
        # A bounded dose is rank-addressed by the downstream assessment: the
        # supported treatment member must occupy the leading proposal rank so
        # the remaining members are an explicit neutral control stratum.  The
        # ordering is derived only from the typed dose contract, remains stable
        # within both strata, and is workload/model/provider independent.
        selected = tuple(
            sorted(
                selected,
                key=lambda option: not bool(cards_by_option[option.option_id]),
            )
        )
        selected_ids = tuple(option.option_id for option in selected)
        members: list[dict[str, object]] = []
        for index, option in enumerate(selected):
            common: dict[str, object] = {
                "supporting_card_keys": list(cards_by_option[option.option_id]),
                "effect_predictions": [
                    {
                        "metric_id": metric_id,
                        "direction": "unknown",
                        "confidence": "unknown",
                    }
                    for metric_id in metric_ids
                ],
                "role_proposal": (
                    "exploit" if index < 4 else "falsify" if index < 7 else "coverage"
                ),
                "design_rationale": (
                    "Outcome-blind provider-free conformance proposal derived "
                    "from the sealed finite action contract."
                ),
            }
            if option.option_id in bindings:
                members.append(
                    {
                        "action_kind": "compose_r2",
                        "composite_option_id": option.option_id,
                        "component_option_ids": list(bindings[option.option_id]),
                        **common,
                    }
                )
            elif bindings:
                members.append(
                    {
                        "action_kind": "atomic",
                        "option_id": option.option_id,
                        **common,
                    }
                )
            else:
                members.append({"option_id": option.option_id, **common})
        value = output_type.model_validate({"members": members}, strict=True)
        self.calls += 1
        self.records.append(
            ProviderFreeCalibratedCallRecord(
                call_id=request.call_id.value,
                proposal_option_ids=selected_ids,
                supporting_card_keys_by_member=tuple(
                    cards_by_option[value] for value in selected_ids
                ),
                assigned_card_keys=tuple(sorted(output_type.assigned_card_keys)),
                evaluation_witness_option_ids=witness,
                required_proposal_support_option_ids=tuple(
                    sorted(output_type.required_proposal_support_option_ids)
                ),
                required_composite_proposals=output_type.required_composite_proposals,
                bounded_memory_dose=output_type.memory_dose_contract is not None,
            ).to_record()
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/calibrated-conformance",
            resolved_model="provider-free/calibrated-conformance",
            resolved_provider="local-deterministic-conformance",
            provider_response_id=None,
            finish_reason="policy_completed",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=0,
        )


__all__ = [
    "ProviderFreeCalibratedCallRecord",
    "ProviderFreeCalibratedConformanceError",
    "ProviderFreeCalibratedPortfolioRunner",
]
