"""Authenticated campaign bridge for the calibrated K8-to-K4 selector.

One coordinator owns the immutable pre-call input binding for each logical
selector request.  The provider adapter, campaign plaintext audit, and strict
post-evaluation prediction decoder all resolve through that same registry.
This prevents a campaign from silently using three independently reconstructed
views of what the model saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_evolve.application.portfolio_evolution import PortfolioVariationWaveResult
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    CalibratedPortfolioAllocator,
    PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy,
    PydanticAIRegretBoundedInformationPortfolioSelectionPolicy,
    PydanticAICalibratedPortfolioSelectionPolicy,
    PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy,
    PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy,
    PydanticAIContextualSearchAllocationPortfolioSelectionPolicy,
    PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy,
    PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy,
    PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy,
    PydanticAIFullSupportCalibratedPortfolioSelectionPolicy,
    PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy,
    PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy,
    PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy,
    PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy,
    PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy,
    decode_calibrated_portfolio_audit,
    render_calibrated_portfolio_selection_prompt_for_allocator,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import LowLevelRunner
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.selection.calibrated_slate import (
    SlateAllocationMode,
    TraceCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlatePolicy,
)
from agent_evolve.policies.selection.frontier_probe_slate import (
    FrontierProbeSlatePolicy,
)
from agent_evolve.policies.selection.full_support_slate import (
    FullSupportSlatePolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    TargetConditionedAllocationContext,
    TargetConditionedAllocationContextKey,
    TargetConditionedAllocationContextRegistrar,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    TargetConditionedSlateDecision,
)
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContextProvider,
    AcquisitionCertifiedSlatePolicy,
)
from agent_evolve.policies.selection.regret_bounded_slate import (
    RegretBoundedSlatePolicy,
)
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    CalibratedPortfolioInputBinding,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastPredictionReceipt,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
)
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.variation_source import finite_variation_source_by_option


@dataclass(frozen=True, slots=True)
class _RegisteredCalibratedPortfolioInput:
    request: PortfolioSelectionRequest
    binding: CalibratedPortfolioInputBinding

    def __post_init__(self) -> None:
        if type(self.request) is not PortfolioSelectionRequest:
            raise TypeError("request must be exact PortfolioSelectionRequest")
        self.request.__post_init__()
        if type(self.binding) is not CalibratedPortfolioInputBinding:
            raise TypeError("binding must be exact CalibratedPortfolioInputBinding")
        self.binding.require_request(self.request)


@dataclass(slots=True)
class CalibratedPortfolioCampaignCoordinator:
    """Single source of truth for calibrated selector campaign evidence.

    A wave factory must call :meth:`register` after it has built the public
    request and before the selector can execute it.  Registration is append
    only: even an identical second registration is rejected, so a later stage
    cannot replace pre-call calibration or structural evidence.

    The object directly satisfies both dependencies needed by a calibrated
    campaign::

        selector = PydanticAICalibratedPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=coordinator.binding_for,
        )
        runtime.selector_request_prompt_renderer = coordinator
    """

    allocator: CalibratedPortfolioAllocator = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    )
    constraint_decoupled: bool = False
    minimum_intervention_projection: bool = False
    evidence_calibrated_source_mix: bool = False
    contextual_search_allocation: bool = False
    _entries: dict[str, _RegisteredCalibratedPortfolioInput] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _decoded_results: dict[str, tuple[PortfolioVariationWaveResult, Any]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.constraint_decoupled) is not bool:
            raise TypeError("constraint_decoupled must be an exact bool")
        if type(self.minimum_intervention_projection) is not bool:
            raise TypeError("minimum_intervention_projection must be an exact bool")
        if type(self.evidence_calibrated_source_mix) is not bool:
            raise TypeError("evidence_calibrated_source_mix must be an exact bool")
        if type(self.contextual_search_allocation) is not bool:
            raise TypeError("contextual_search_allocation must be an exact bool")
        if self.contextual_search_allocation and not (
            self.evidence_calibrated_source_mix
            and self.minimum_intervention_projection
            and self.constraint_decoupled
        ):
            raise ValueError(
                "contextual search allocation requires evidence-calibrated "
                "source mix, minimum intervention, and constraint-decoupled "
                "authority"
            )
        if (
            self.evidence_calibrated_source_mix
            and not self.minimum_intervention_projection
        ):
            raise ValueError(
                "evidence-calibrated source mix requires minimum intervention"
            )
        if self.minimum_intervention_projection and not self.constraint_decoupled:
            raise ValueError(
                "minimum intervention requires constraint-decoupled authority"
            )
        if self.constraint_decoupled and type(self.allocator) not in {
            HorizonBoundedStructuralPosteriorSlatePolicy,
            TargetConditionedSlateAllocatorAdapter,
            AcquisitionCertifiedSlatePolicy,
            RegretBoundedSlatePolicy,
        }:
            raise TypeError(
                "constraint-decoupled acquisition requires the horizon-bounded "
                "target-conditioned, acquisition-certified, or regret-bounded "
                "allocator"
            )
        if (
            type(self.allocator) is TargetConditionedSlateAllocatorAdapter
            and (
                self.minimum_intervention_projection
                or self.evidence_calibrated_source_mix
                or self.contextual_search_allocation
            )
        ):
            raise ValueError(
                "target-conditioned constraint decoupling does not imply the "
                "horizon-only projection or source-mix treatments"
            )
        if type(self.allocator) is TraceCalibratedSlatePolicy:
            self.allocator.__post_init__()
            if self.allocator.mode is not SlateAllocationMode.CALIBRATED_FOUR_ROLE:
                raise ValueError(
                    "campaign coordinator requires calibrated four-role mode"
                )
            return
        if type(self.allocator) is ModelAnchoredCalibratedSlatePolicy:
            self.allocator.__post_init__()
            return
        if type(self.allocator) is StructuralPosteriorSlatePolicy:
            self.allocator.__post_init__()
            return
        if type(self.allocator) is OperatorStratifiedStructuralPosteriorSlatePolicy:
            self.allocator.__post_init__()
            return
        if type(self.allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
            self.allocator.__post_init__()
            return
        if type(self.allocator) is FrontierProbeSlatePolicy:
            self.allocator.__post_init__()
            return
        if type(self.allocator) is FullSupportSlatePolicy:
            return
        if type(self.allocator) is TargetConditionedSlateAllocatorAdapter:
            self.allocator.__post_init__()
            if not isinstance(
                self.allocator.context_provider,
                TargetConditionedAllocationContextRegistrar,
            ):
                raise TypeError(
                    "campaign target-conditioned context provider must support "
                    "append-only registration"
                )
            return
        if type(self.allocator) in {
            AcquisitionCertifiedSlatePolicy,
            RegretBoundedSlatePolicy,
        }:
            self.allocator.__post_init__()
            if not isinstance(
                self.allocator.context_provider,
                AcquisitionCertifiedSlateContextProvider,
            ):
                raise TypeError(
                    "campaign acquisition-certified provider must expose contexts"
                )
            return
        raise TypeError("allocator must be an exact supported calibrated policy")

    def build_selector(
        self,
        generate_once: LowLevelRunner,
    ) -> (
        PydanticAICalibratedPortfolioSelectionPolicy
        | PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy
        | PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy
        | PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
        | PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy
        | PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy
        | PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy
        | PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy
        | PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy
        | PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy
        | PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy
        | PydanticAIFullSupportCalibratedPortfolioSelectionPolicy
        | PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy
        | PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy
        | PydanticAIRegretBoundedInformationPortfolioSelectionPolicy
    ):
        """Build a selector sharing this coordinator's trusted allocator."""

        self.__post_init__()
        if self.constraint_decoupled:
            if type(self.allocator) is AcquisitionCertifiedSlatePolicy:
                return PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy(
                    generate_once=generate_once,
                    binding_for=self.binding_for,
                    allocator=self.allocator,
                )
            if type(self.allocator) is RegretBoundedSlatePolicy:
                return PydanticAIRegretBoundedInformationPortfolioSelectionPolicy(
                    generate_once=generate_once,
                    binding_for=self.binding_for,
                    allocator=self.allocator,
                )
            if type(self.allocator) is TargetConditionedSlateAllocatorAdapter:
                return (
                    PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy(
                        generate_once=generate_once,
                        binding_for=self.binding_for,
                        allocator=self.allocator,
                    )
                )
            assert type(self.allocator) is HorizonBoundedStructuralPosteriorSlatePolicy
            if self.minimum_intervention_projection:
                if self.contextual_search_allocation:
                    return PydanticAIContextualSearchAllocationPortfolioSelectionPolicy(
                        generate_once=generate_once,
                        binding_for=self.binding_for,
                        allocator=self.allocator,
                    )
                if self.evidence_calibrated_source_mix:
                    return (
                        PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy(
                            generate_once=generate_once,
                            binding_for=self.binding_for,
                            allocator=self.allocator,
                        )
                    )
                return PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy(
                    generate_once=generate_once,
                    binding_for=self.binding_for,
                    allocator=self.allocator,
                )
            return PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is TraceCalibratedSlatePolicy:
            return PydanticAICalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is ModelAnchoredCalibratedSlatePolicy:
            return PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is StructuralPosteriorSlatePolicy:
            return PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is OperatorStratifiedStructuralPosteriorSlatePolicy:
            return PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
            return PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is FrontierProbeSlatePolicy:
            return PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is FullSupportSlatePolicy:
            return PydanticAIFullSupportCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        if type(self.allocator) is TargetConditionedSlateAllocatorAdapter:
            return PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy(
                generate_once=generate_once,
                binding_for=self.binding_for,
                allocator=self.allocator,
            )
        raise AssertionError("coordinator allocator changed after validation")

    def register(
        self,
        request: PortfolioSelectionRequest,
        binding: CalibratedPortfolioInputBinding,
        *,
        target_conditioned_context: TargetConditionedAllocationContext | None = None,
    ) -> None:
        """Seal exactly one authenticated binding for ``request``."""

        entry = _RegisteredCalibratedPortfolioInput(request, binding)
        request_sha256 = request.request_sha256
        if request_sha256 in self._entries:
            raise ValueError("calibrated campaign request is already registered")
        if type(self.allocator) is TargetConditionedSlateAllocatorAdapter:
            if type(target_conditioned_context) is not (
                TargetConditionedAllocationContext
            ):
                raise TypeError(
                    "target-conditioned campaigns require one exact pre-call context"
                )
            registrar = self.allocator.context_provider
            if not isinstance(registrar, TargetConditionedAllocationContextRegistrar):
                raise TypeError("target-conditioned context provider cannot register")
            observed_transitions = tuple(
                (
                    transition.option_id,
                    transition.option_identity_sha256,
                    transition.parent_configuration_sha256,
                    transition.child_configuration_sha256,
                )
                for transition in target_conditioned_context.transition_receipts
            )
            if observed_transitions:
                expected_transitions = tuple(
                    (
                        option.option_id,
                        option.identity_sha256,
                        option.parent_configuration_sha256,
                        option.child_configuration_sha256,
                    )
                    for option in sorted(
                        request.finite_variation_contract.options,
                        key=lambda value: value.option_id,
                    )
                )
                if observed_transitions != expected_transitions:
                    raise ValueError(
                        "precomputed target-conditioned transitions must exactly "
                        "cover the finite variation contract"
                    )
            registrar.register(
                TargetConditionedAllocationContextKey(
                    scope_sha256=binding.context.scope.scope_sha256,
                    wave_index=binding.context.wave_index,
                    parent_configuration_sha256=(
                        binding.context.parent_candidate_identity_sha256
                    ),
                    finite_contract_sha256=(
                        request.finite_variation_contract.identity_sha256
                    ),
                ),
                target_conditioned_context,
                request.finite_variation_contract,
            )
        elif target_conditioned_context is not None:
            raise ValueError(
                "target_conditioned_context is only valid for its exact allocator"
            )
        self._entries[request_sha256] = entry

    def _entry_for(
        self,
        request: PortfolioSelectionRequest,
    ) -> _RegisteredCalibratedPortfolioInput:
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be exact PortfolioSelectionRequest")
        request.__post_init__()
        entry = self._entries.get(request.request_sha256)
        if entry is None:
            raise ValueError("calibrated campaign request is foreign or unregistered")
        entry.__post_init__()
        if request != entry.request:
            raise ValueError("calibrated campaign request differs from registration")
        return entry

    def binding_for(
        self,
        request: PortfolioSelectionRequest,
    ) -> CalibratedPortfolioInputBinding:
        """Supply the selector only its prospectively registered binding."""

        entry = self._entry_for(request)
        entry.binding.require_request(request)
        return entry.binding

    def render(self, request: PortfolioSelectionRequest) -> str:
        """Render the exact calibrated provider prompt from the sealed binding."""

        return render_calibrated_portfolio_selection_prompt_for_allocator(
            request,
            self.binding_for(request),
            self.allocator,
            constraint_decoupled=self.constraint_decoupled,
            minimum_intervention_projection=(self.minimum_intervention_projection),
            evidence_calibrated_source_mix=(self.evidence_calibrated_source_mix),
            contextual_search_allocation=self.contextual_search_allocation,
        )

    def _decode_result(self, result: PortfolioVariationWaveResult) -> Any:
        """Strictly replay one immutable wave result at most once per instance."""

        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("result must be exact PortfolioVariationWaveResult")
        result.__post_init__()
        key = result.receipt.receipt_sha256
        cached = self._decoded_results.get(key)
        if cached is not None and cached[0] is result:
            return cached[1]
        entry = self._entries.get(result.receipt.request_sha256)
        if entry is None:
            raise ValueError("portfolio result belongs to a foreign campaign request")
        entry.__post_init__()
        if result.receipt.selection_call_id != entry.request.call_id:
            raise ValueError("portfolio result names a foreign selector call")
        audit = result.supplemental_selection_audit
        if audit is None:
            raise ValueError("calibrated portfolio result omitted supplemental audit")
        decoded = decode_calibrated_portfolio_audit(
            audit,
            request=entry.request,
            binding=entry.binding,
            allocator=self.allocator,
        )
        self._decoded_results[key] = (result, decoded)
        return decoded

    def decode_target_conditioned_allocation(
        self,
        result: PortfolioVariationWaveResult,
    ) -> TargetConditionedSlateDecision:
        """Strictly replay one result and return its exact T-RAP decision."""

        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("result must be exact PortfolioVariationWaveResult")
        result.__post_init__()
        entry = self._entries.get(result.receipt.request_sha256)
        if entry is None:
            raise ValueError("portfolio result belongs to a foreign campaign request")
        entry.__post_init__()
        if type(self.allocator) is not TargetConditionedSlateAllocatorAdapter:
            raise ValueError("campaign allocator is not target-conditioned")
        if result.receipt.selection_call_id != entry.request.call_id:
            raise ValueError("portfolio result names a foreign selector call")
        audit = result.supplemental_selection_audit
        if audit is None:
            raise ValueError("target-conditioned result omitted supplemental audit")
        decoded = self._decode_result(result)
        allocation = decoded.allocation
        if type(allocation) is not TargetConditionedSlateDecision:
            raise TypeError("decoded allocation is not an exact T-RAP decision")
        selected = tuple(value.option_id for value in allocation.selected)
        evaluated = tuple(
            value.materialization.option_id for value in result.receipt.members
        )
        if selected != evaluated:
            raise ValueError("decoded T-RAP decision differs from evaluated members")
        return allocation

    def decode_target_conditioned_context(
        self,
        result: PortfolioVariationWaveResult,
    ) -> TargetConditionedAllocationContext:
        """Recover the independently registered pre-call context after replay."""

        allocation = self.decode_target_conditioned_allocation(result)
        assert type(self.allocator) is TargetConditionedSlateAllocatorAdapter
        context = self.allocator.context_provider.context_for(
            allocation.request.allocation_request
        )
        if type(context) is not TargetConditionedAllocationContext:
            raise TypeError("registered provider returned a foreign T-RAP context")
        context.__post_init__()
        if (
            context.frontier_target != allocation.request.frontier_target
            or context.state != allocation.request.state
        ):
            raise ValueError("registered context differs from decoded T-RAP decision")
        return context

    def decode_selected_predictions(
        self,
        result: PortfolioVariationWaveResult,
    ) -> tuple[ForecastPredictionReceipt, ...]:
        """Strictly replay one result and return only its evaluated K4 forecasts."""

        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("result must be exact PortfolioVariationWaveResult")
        result.__post_init__()
        entry = self._entries.get(result.receipt.request_sha256)
        if entry is None:
            raise ValueError("portfolio result belongs to a foreign campaign request")
        entry.__post_init__()
        request = entry.request
        if result.receipt.selection_call_id != request.call_id:
            raise ValueError("portfolio result names a foreign selector call")
        audit = result.supplemental_selection_audit
        if audit is None:
            raise ValueError("calibrated portfolio result omitted supplemental audit")
        decoded = self._decode_result(result)
        predictions = decoded.selected_prediction_receipts
        selected_option_ids = tuple(
            member.materialization.option_id for member in result.receipt.members
        )
        predicted_option_ids = tuple(
            dict.fromkeys(value.option_id for value in predictions)
        )
        if predicted_option_ids != selected_option_ids:
            raise ValueError(
                "decoded forecasts differ from evaluated portfolio member order"
            )
        if len(predictions) != (
            len(selected_option_ids) * len(request.required_metric_ids)
        ):
            raise ValueError("decoded forecasts do not exactly cover evaluated metrics")
        return predictions

    def decode_selected_source_ids(
        self,
        result: PortfolioVariationWaveResult,
    ) -> tuple[str, ...]:
        """Replay one result and return sealed finite-variation source labels.

        Model retention versus deterministic engine insertion remains in the
        reconciliation receipt as a separate provenance axis.  Source credit
        instead follows the option's prospectively sealed catalogue metadata,
        with ordinary local options inheriting the framework ``primary`` arm.
        No outcome, workload, model-route, or provider field participates.
        """

        if type(result) is not PortfolioVariationWaveResult:
            raise TypeError("result must be exact PortfolioVariationWaveResult")
        result.__post_init__()
        entry = self._entries.get(result.receipt.request_sha256)
        if entry is None:
            raise ValueError("portfolio result belongs to a foreign campaign request")
        entry.__post_init__()
        audit = result.supplemental_selection_audit
        if audit is None:
            raise ValueError("calibrated portfolio result omitted supplemental audit")
        # Exact replay authenticates the complete payload before the source
        # projection below reads its reconciliation evidence.
        self._decode_result(result)
        payload = thaw_json(audit.payload)
        if type(payload) is not dict:
            raise AssertionError("calibrated audit payload did not thaw to an object")
        reconciliation = payload.get("semantic_reconciliation")
        selected = tuple(
            member.materialization.option_id for member in result.receipt.members
        )
        if reconciliation is None:
            if self.constraint_decoupled:
                raise ValueError("constraint-decoupled source decode needs reconciliation")
        elif type(reconciliation) is not dict:
            raise TypeError("semantic reconciliation must be an exact object")
        else:
            members = reconciliation.get("members")
            if type(members) is not list:
                raise TypeError("semantic reconciliation omitted members")
            reconciled_option_ids = tuple(
                member.get("option_id")
                for member in members
                if type(member) is dict and type(member.get("option_id")) is str
            )
            if len(reconciled_option_ids) != len(members) or len(
                set(reconciled_option_ids)
            ) != len(reconciled_option_ids):
                raise ValueError(
                    "semantic reconciliation option identities are invalid"
                )
            if not set(selected).issubset(reconciled_option_ids):
                raise ValueError("selected portfolio escapes semantic reconciliation")
        source_by_option = finite_variation_source_by_option(
            entry.request.finite_variation_contract
        )
        if not set(selected).issubset(source_by_option):
            raise ValueError("selected portfolio escapes finite source attribution")
        return tuple(source_by_option[option_id] for option_id in selected)

    def decode_contextual_allocation_realization(
        self,
        result: PortfolioVariationWaveResult | PortfolioSelectionResult,
    ) -> ContextualPortfolioAllocationRealization | None:
        """Project an authenticated selector receipt into the generic port.

        The calibrated integration owns finite-set reconciliation.  Only its
        requested and realized arm counts cross the inverted API; candidate
        fields and workload/model/provider identifiers remain private here.
        """

        if type(result) is PortfolioVariationWaveResult:
            result.__post_init__()
            request_sha256 = result.receipt.request_sha256
            audit = result.supplemental_selection_audit
        elif type(result) is PortfolioSelectionResult:
            result.__post_init__()
            request_sha256 = result.decision.request_sha256
            audit = result.supplemental_audit
        else:
            raise TypeError(
                "result must be exact PortfolioVariationWaveResult or "
                "PortfolioSelectionResult"
            )
        entry = self._entries.get(request_sha256)
        if entry is None:
            raise ValueError("portfolio result belongs to a foreign campaign request")
        entry.__post_init__()
        if audit is None:
            raise ValueError("calibrated portfolio result omitted supplemental audit")
        decode_calibrated_portfolio_audit(
            audit,
            request=entry.request,
            binding=entry.binding,
            allocator=self.allocator,
        )
        contract = entry.binding.contextual_allocation
        if contract is None:
            return None
        payload = thaw_json(audit.payload)
        if type(payload) is not dict:
            raise AssertionError("calibrated audit payload did not thaw to an object")
        reconciliation = payload.get("semantic_reconciliation")
        if type(reconciliation) is not dict:
            raise ValueError("contextual selection omitted semantic reconciliation")
        projection = reconciliation.get("contextual_allocation_projection")
        if type(projection) is not dict:
            raise ValueError("contextual selection omitted allocation projection")

        def counts(name: str) -> tuple[tuple[str, int], ...]:
            value = projection.get(name)
            if type(value) is not list:
                raise TypeError(f"{name} must be an exact list")
            rows = tuple(
                (row[0], row[1]) for row in value if type(row) is list and len(row) == 2
            )
            if len(rows) != len(value):
                raise TypeError(f"{name} contains a malformed count")
            return rows

        realization = ContextualPortfolioAllocationRealization(
            campaign_scope_sha256=contract.campaign_scope_sha256,
            query_sha256=contract.query_sha256,
            decision_sha256=contract.decision_sha256,
            contract_sha256=contract.contract_sha256,
            controller_wave_index=contract.controller_wave_index,
            slice_id=contract.slice_id,
            requested_source_target_counts=counts("requested_source_target_counts"),
            requested_operator_target_counts=counts("requested_operator_target_counts"),
            realized_source_target_counts=counts("realized_source_target_counts"),
            realized_operator_target_counts=counts("realized_operator_target_counts"),
            requested_minimum_single_path_interventions=(
                contract.minimum_single_path_interventions
            ),
            realized_single_path_interventions=(
                projection.get("realized_single_path_interventions", 0)
            ),
            requested_minimum_disjoint_parent_patch_pairs=(
                contract.minimum_disjoint_parent_patch_pairs
            ),
            realized_disjoint_parent_patch_pairs=(
                projection.get("realized_disjoint_parent_patch_pairs", 0)
            ),
        )
        realization.require_contract(contract)
        if projection.get("source_l1_deviation") != realization.source_l1_deviation:
            raise ValueError("source allocation projection L1 drifted")
        if projection.get("operator_l1_deviation") != (
            realization.operator_l1_deviation
        ):
            raise ValueError("operator allocation projection L1 drifted")
        if projection.get("exact") is not realization.exact:
            raise ValueError("allocation projection exact verdict drifted")
        if projection.get("objective_values_consulted") is not False:
            raise ValueError("allocation projection consulted objective values")
        if projection.get("workload_identifiers_consulted") is not False:
            raise ValueError("allocation projection consulted workload identifiers")
        return realization

    @property
    def registered_request_count(self) -> int:
        """Number of append-only pre-call request bindings."""

        return len(self._entries)


__all__ = ["CalibratedPortfolioCampaignCoordinator"]
