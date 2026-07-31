from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    BrokerEvidenceChannel,
    BrokerReturnEstimate,
    EmpiricalBayesMaterializedActionReturnValue,
    MaterializedActionBrokerRequest,
    MaterializedActionContext,
    MaterializedActionDelayedCredit,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    MaterializedActionOpportunityEvidence,
    MaterializedActionOutcome,
    MaterializedActionResolvedReturn,
    MaterializedActionReturnPriorPrediction,
    OpportunityConditionedMaterializedActionReturnValue,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionAllocationDirective,
)
from agent_evolve.application.prequential_residual_exploration import (
    PrequentialLowDiscrepancyResidualExploration,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json, thaw_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _context(
    decision_index: int,
    *,
    phase: SearchPhase = SearchPhase.BASIN_EXPANSION,
    cell: str = "frontier_a",
) -> MaterializedActionContext:
    return MaterializedActionContext(
        campaign_scope_sha256=_sha("campaign"),
        decision_index=decision_index,
        phase=phase,
        remaining_decisions=3,
        remaining_evaluations=12,
        residual_frontier_cell=cell,
        parent_position_cell="parent_edge",
        archive_relation_cell="nondominated_near",
        structural_signature_sha256=_sha("structure"),
        patch_compatibility_cell="compatible",
        forecast_calibration_cell="medium_support",
        source_distance_bin=1,
        memory_dose_bin=0,
    )


def _action(
    name: str,
    *,
    context: MaterializedActionContext,
    expert: str,
    operator: str,
    arity: int,
    rank: int = 1,
    reference: bool = False,
    phenotype: str | None = None,
) -> MaterializedActionDescriptor:
    configuration = freeze_json({"candidate": name})
    return MaterializedActionDescriptor(
        context=context,
        configuration=configuration,
        phenotype_identity_sha256=_sha(phenotype or f"phenotype:{name}"),
        expert_id=expert,
        native_rank=rank,
        parent_ids=tuple(
            CandidateId(f"candidate_parent_{name}_{index}")
            for index in range(arity)
        ),
        operator_id=operator,
        target_candidate_id=CandidateId(f"candidate_target_{name}"),
        role_id="frontier_candidate",
        normalized_evaluation_cost=0.5,
        reference_action=reference,
    )


@dataclass(frozen=True)
class _AlwaysFeasible:
    definition_sha256: str = _sha("always-feasible")

    def permits(self, actions):
        return True


@dataclass(frozen=True)
class _SynergyValue:
    preferred_names: frozenset[str]
    definition_sha256: str = _sha("synergy-value")

    def value(self, actions):
        if not self.preferred_names:
            return 0.0
        names = {
            dict(value.configuration.items)["candidate"] for value in actions
        }
        return 1.0 if self.preferred_names.issubset(names) else 0.0


@dataclass(frozen=True)
class _PortableReturnValue:
    preferred_name: str
    definition_sha256: str = _sha("portable-return-value")

    def estimate(self, action):
        name = dict(action.configuration.items)["candidate"]
        mean = 0.8 if name == self.preferred_name else 0.1
        return BrokerReturnEstimate(
            mean=mean,
            standard_deviation=0.05,
            local_count=0,
            global_count=0,
            resolved_count=0,
            provisional_count=0,
            local_mean=mean,
            global_mean=mean,
            shrinkage_weight=0.0,
        )


@dataclass(frozen=True)
class _PortableReturnPrior:
    preferred_name: str
    definition_sha256: str = _sha("portable-return-prior")

    def predict(self, action):
        name = dict(action.configuration.items)["candidate"]
        return MaterializedActionReturnPriorPrediction(
            action_sha256=action.action_sha256,
            mean=0.8 if name == self.preferred_name else 0.2,
            standard_deviation=0.1,
            effective_sample_size=1.0,
            evidence_sha256=_sha("held-out-cross-run-panel"),
        )


@dataclass(frozen=True)
class _FlatReturnValue:
    definition_sha256: str = _sha("flat-return-value")

    def estimate(self, action):
        return BrokerReturnEstimate(
            mean=0.2,
            standard_deviation=0.5,
            local_count=0,
            global_count=0,
            resolved_count=0,
            provisional_count=0,
            local_mean=0.2,
            global_mean=0.2,
            shrinkage_weight=0.0,
        )


@dataclass(frozen=True)
class _ParentOpportunity:
    preferred_name: str
    mismatch: bool = False
    definition_sha256: str = _sha("parent-opportunity")

    def estimate(self, action):
        name = dict(action.configuration.items)["candidate"]
        action_sha256 = (
            _sha("wrong-action")
            if self.mismatch
            else action.action_sha256
        )
        return MaterializedActionOpportunityEvidence(
            action_sha256=action_sha256,
            source_opportunity=0.1 if name == self.preferred_name else 0.0,
            archive_opportunity_scale=0.1,
            evidence_sha256=_sha(f"opportunity:{name}"),
        )


def test_prior_outcomes_reallocate_budget_away_from_zero_yield_parent_arity():
    ledger = MaterializedActionEvidenceLedger()
    prior_context = _context(1)
    for index in range(4):
        mutation = _action(
            f"prior_mutation_{index}",
            context=prior_context,
            expert="semantic",
            operator="mutation",
            arity=1,
        )
        crossover = _action(
            f"prior_crossover_{index}",
            context=prior_context,
            expert="composition",
            operator="crossover",
            arity=2,
        )
        ledger.append_outcome(MaterializedActionOutcome(mutation, True, True, 0.8, True))
        ledger.append_outcome(MaterializedActionOutcome(crossover, True, True, 0.0, False))

    current = _context(2)
    mutation = _action(
        "current_mutation",
        context=current,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    crossover = _action(
        "current_crossover",
        context=current,
        expert="composition",
        operator="crossover",
        arity=2,
    )
    policy = RegretBrokeredMaterializedActionPolicy(ledger)
    decision = policy.select(
        MaterializedActionBrokerRequest(
            actions=(mutation, crossover),
            evaluation_slots=1,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    assert decision.selected_actions == (mutation,)
    scores = {value.action_sha256: value for value in decision.scores}
    assert scores[mutation.action_sha256].value > scores[crossover.action_sha256].value


def test_empirical_bayes_return_value_starts_from_prior_then_yields_to_real_data():
    ledger = MaterializedActionEvidenceLedger()
    prior_context = _context(1)
    for index in range(10):
        action = _action(
            f"historical_preferred_{index}",
            context=prior_context,
            expert="semantic",
            operator="mutation",
            arity=1,
        )
        ledger.append_outcome(
            MaterializedActionOutcome(action, True, True, 0.0, False)
        )

    current = _context(2)
    preferred = _action(
        "preferred",
        context=current,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    alternative = _action(
        "alternative",
        context=current,
        expert="coverage",
        operator="restart",
        arity=0,
    )
    value = EmpiricalBayesMaterializedActionReturnValue(
        ledger=ledger,
        prior=_PortableReturnPrior(preferred_name="preferred"),
    )

    preferred_estimate = value.estimate(preferred)
    alternative_estimate = value.estimate(alternative)
    assert preferred_estimate.global_count == 10
    assert preferred_estimate.mean < alternative_estimate.mean
    assert alternative_estimate.global_count == 0
    assert alternative_estimate.mean == 0.2


def test_opportunity_conditioning_revalues_parent_and_caps_unit_uncertainty():
    context = _context(1)
    preferred = _action(
        "preferred_parent",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    alternative = _action(
        "alternative_parent",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    value = OpportunityConditionedMaterializedActionReturnValue(
        base=_FlatReturnValue(),
        opportunity=_ParentOpportunity(preferred_name="preferred_parent"),
    )

    preferred_estimate = value.estimate(preferred)
    alternative_estimate = value.estimate(alternative)
    assert preferred_estimate.mean == 0.4
    assert preferred_estimate.standard_deviation == 0.4
    assert alternative_estimate.mean == 0.2
    assert alternative_estimate.standard_deviation == 0.2

    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger(),
        return_value=value,
    ).select(
        MaterializedActionBrokerRequest(
            actions=(alternative, preferred),
            evaluation_slots=1,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )
    assert decision.selected_actions == (preferred,)


def test_opportunity_evidence_rejects_foreign_action_identity():
    action = _action(
        "foreign",
        context=_context(1),
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    value = OpportunityConditionedMaterializedActionReturnValue(
        base=_FlatReturnValue(),
        opportunity=_ParentOpportunity(
            preferred_name="foreign",
            mismatch=True,
        ),
    )

    try:
        value.estimate(action)
    except ValueError as error:
        assert "another action" in str(error)
    else:
        raise AssertionError("foreign opportunity identity was accepted")


def test_delayed_descendant_credit_stays_separate_from_immediate_gain():
    ledger = MaterializedActionEvidenceLedger()
    action = _action(
        "delayed",
        context=_context(1),
        expert="restart",
        operator="global_restart",
        arity=0,
    )
    outcome = MaterializedActionOutcome(action, True, True, 0.0, False)
    ledger.append_outcome(outcome)
    ledger.append_delayed_credit(
        MaterializedActionDelayedCredit(
            outcome=outcome,
            available_at_decision_index=2,
            stage_front_survived=False,
            useful_descendant_observed=True,
        )
    )

    score = RegretBrokeredMaterializedActionPolicy(ledger).score(
        _action(
            "future_restart",
            context=_context(2),
            expert="restart",
            operator="global_restart",
            arity=0,
        )
    )
    estimates = {value.channel: value for value in score.estimates}
    assert estimates[BrokerEvidenceChannel.GAIN].mean < 0.5
    assert estimates[BrokerEvidenceChannel.DESCENDANT].mean > 0.5


def test_resolved_lineage_return_replaces_provisional_immediate_return():
    ledger = MaterializedActionEvidenceLedger()
    source = _action(
        "resolved_source",
        context=_context(1),
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    outcome = MaterializedActionOutcome(source, True, True, 0.2, True)
    ledger.append_outcome(outcome)
    ledger.append_resolved_return(
        MaterializedActionResolvedReturn(
            outcome=outcome,
            available_at_decision_index=2,
            horizon_end_decision_index=1,
            normalized_immediate_return=0.2,
            normalized_descendant_return=0.5,
            normalized_horizon_return=0.7,
            fully_resolved=True,
            attribution_definition_sha256=_sha("equal-lineage-attribution"),
        )
    )

    score = RegretBrokeredMaterializedActionPolicy(ledger).score(
        _action(
            "future_semantic",
            context=_context(3),
            expert="semantic",
            operator="mutation",
            arity=1,
        )
    )

    assert score.return_estimate.resolved_count == 1
    assert score.return_estimate.provisional_count == 0
    assert score.return_estimate.global_mean > 0.5


def test_terminal_score_has_no_information_bonus():
    policy = RegretBrokeredMaterializedActionPolicy(MaterializedActionEvidenceLedger())
    nonterminal = policy.score(
        _action(
            "nonterminal",
            context=_context(1, phase=SearchPhase.BASIN_ACQUISITION),
            expert="semantic",
            operator="mutation",
            arity=1,
        )
    )
    terminal = policy.score(
        _action(
            "terminal",
            context=_context(1, phase=SearchPhase.TERMINAL_CONVERSION),
            expert="semantic",
            operator="mutation",
            arity=1,
        )
    )

    assert nonterminal.selection_index > nonterminal.value
    assert terminal.selection_index == terminal.value


def test_one_nonterminal_slate_purchases_at_most_one_information_action():
    context = _context(1, phase=SearchPhase.BASIN_EXPANSION)
    actions = tuple(
        _action(
            f"unseen_{index}",
            context=context,
            expert=f"expert_{index}",
            operator="mutation",
            arity=1,
        )
        for index in range(3)
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    ).select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=3,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    assert all(value.value == 0.0 for value in decision.scores)
    assert decision.exploration_action_sha256 in {
        value.action_sha256 for value in decision.selected_actions
    }


def test_terminal_slate_records_no_exploration_action():
    context = _context(1, phase=SearchPhase.TERMINAL_CONVERSION)
    actions = tuple(
        _action(
            f"terminal_{index}",
            context=context,
            expert=f"expert_{index}",
            operator="mutation",
            arity=1,
        )
        for index in range(2)
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    ).select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=2,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    assert decision.exploration_action_sha256 is None


def test_injected_portable_return_value_controls_unseen_cross_expert_choice():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    numerical = _action(
        "numerical",
        context=context,
        expert="numerical",
        operator="global",
        arity=0,
    )
    semantic = _action(
        "semantic",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    value = _PortableReturnValue(preferred_name="semantic")
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger(),
        return_value=value,
    ).select(
        MaterializedActionBrokerRequest(
            actions=(numerical, semantic),
            evaluation_slots=1,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    assert decision.selected_actions == (semantic,)
    assert all(
        score.return_estimator_definition_sha256 == value.definition_sha256
        for score in decision.scores
    )


def test_empirical_cold_start_preserves_native_rank_before_hash_order():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    rank_two = _action(
        "rank_two",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
        rank=2,
    )
    rank_one = _action(
        "rank_one",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
        rank=1,
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    ).select(
        MaterializedActionBrokerRequest(
            actions=(rank_two, rank_one),
            evaluation_slots=1,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    assert decision.selected_actions == (rank_one,)


def test_prequential_exploration_covers_lanes_and_an_interior_cold_rank():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    agentic = tuple(
        _action(
            f"agentic_{rank}",
            context=context,
            expert="agentic",
            operator="mutation",
            arity=1,
            rank=rank,
        )
        for rank in range(1, 7)
    )
    recombination = tuple(
        _action(
            f"recombination_{rank}",
            context=context,
            expert="recombination",
            operator="crossover",
            arity=2,
            rank=rank,
        )
        for rank in range(1, 7)
    )
    reference = _action(
        "reference",
        context=context,
        expert="numerical",
        operator="global",
        arity=0,
        reference=True,
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger(),
        exploration_policy=(
            PrequentialLowDiscrepancyResidualExploration()
        ),
    ).select(
        MaterializedActionBrokerRequest(
            actions=(*agentic, *recombination, reference),
            evaluation_slots=4,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=1,
        )
    )

    requirement = decision.exploration_requirement
    assert requirement is not None
    assert requirement.cold_start is True
    assert len(requirement.required_action_sha256s) == 2
    required = {
        value.action_sha256: value
        for value in (*agentic, *recombination)
        if value.action_sha256 in requirement.required_action_sha256s
    }
    assert {
        value.expert_id for value in required.values()
    } == {"agentic", "recombination"}
    assert {
        value.native_rank
        for value in required.values()
        if value.expert_id == "agentic"
    } == {1}
    assert {
        value.native_rank
        for value in required.values()
        if value.expert_id == "recombination"
    } == {1}
    assert reference in decision.selected_actions


def test_prequential_exploration_cycles_one_protected_rank_after_cold_start():
    ledger = MaterializedActionEvidenceLedger()
    prior = _action(
        "prior",
        context=_context(1),
        expert="agentic",
        operator="mutation",
        arity=1,
    )
    ledger.append_outcome(
        MaterializedActionOutcome(prior, True, True, 0.1, True)
    )
    context = _context(2, phase=SearchPhase.BASIN_ACQUISITION)
    agentic = tuple(
        _action(
            f"current_agentic_{rank}",
            context=context,
            expert="agentic",
            operator="mutation",
            arity=1,
            rank=rank,
        )
        for rank in range(1, 7)
    )
    reference = _action(
        "current_reference",
        context=context,
        expert="numerical",
        operator="global",
        arity=0,
        reference=True,
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        ledger,
        exploration_policy=(
            PrequentialLowDiscrepancyResidualExploration()
        ),
    ).select(
        MaterializedActionBrokerRequest(
            actions=(*agentic, reference),
            evaluation_slots=3,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=1,
        )
    )

    requirement = decision.exploration_requirement
    assert requirement is not None
    assert requirement.cold_start is False
    assert requirement.prior_outcome_count == 1
    assert requirement.required_action_sha256s == (
        agentic[3].action_sha256,
    )
    assert agentic[3] in decision.selected_actions


def test_prequential_exploration_suppresses_information_only_terminal_slot():
    ledger = MaterializedActionEvidenceLedger()
    prior = _action(
        "prior_terminal",
        context=_context(1),
        expert="agentic",
        operator="mutation",
        arity=1,
    )
    ledger.append_outcome(
        MaterializedActionOutcome(prior, True, True, 0.1, True)
    )
    context = _context(2, phase=SearchPhase.TERMINAL_CONVERSION)
    actions = tuple(
        _action(
            f"terminal_{rank}",
            context=context,
            expert="agentic",
            operator="mutation",
            arity=1,
            rank=rank,
        )
        for rank in range(1, 7)
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        ledger,
        exploration_policy=(
            PrequentialLowDiscrepancyResidualExploration()
        ),
    ).select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=2,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
        )
    )

    requirement = decision.exploration_requirement
    assert requirement is not None
    assert requirement.cold_start is False
    assert requirement.required_action_sha256s == ()
    evidence = thaw_json(requirement.evidence)
    assert evidence["terminal_information_purchase_suppressed"] is True


def test_prequential_exploration_excludes_required_reference_phenotype():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    reference = _action(
        "reference_duplicate_owner",
        context=context,
        expert="numerical",
        operator="global",
        arity=0,
        reference=True,
        phenotype="shared_phenotype",
    )
    duplicate = _action(
        "challenger_duplicate",
        context=context,
        expert="agentic",
        operator="mutation",
        arity=1,
        phenotype="shared_phenotype",
    )
    distinct = _action(
        "challenger_distinct",
        context=context,
        expert="agentic",
        operator="mutation",
        arity=1,
        rank=2,
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger(),
        exploration_policy=(
            PrequentialLowDiscrepancyResidualExploration()
        ),
    ).select(
        MaterializedActionBrokerRequest(
            actions=(reference, duplicate, distinct),
            evaluation_slots=2,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=1,
        )
    )

    requirement = decision.exploration_requirement
    assert requirement is not None
    assert requirement.required_action_sha256s == ()
    evidence = thaw_json(requirement.evidence)
    assert evidence["challenger_capacity"] == 1
    assert evidence["unreserved_challenger_slots"] == 1
    assert evidence["explorable_capacity"] == 0
    assert decision.selected_actions == tuple(
        sorted((reference, distinct), key=lambda value: value.action_sha256)
    )


def test_joint_slate_escrow_and_phenotype_dedup_are_enforced_together():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    reference = _action(
        "reference",
        context=context,
        expert="reference",
        operator="global",
        arity=0,
        reference=True,
    )
    synergistic = _action(
        "synergistic",
        context=context,
        expert="semantic",
        operator="mutation",
        arity=1,
    )
    duplicate_route = _action(
        "duplicate_route",
        context=context,
        expert="numerical",
        operator="global",
        arity=0,
        phenotype="phenotype:synergistic",
    )
    distractor = _action(
        "distractor",
        context=context,
        expert="coverage",
        operator="restart",
        arity=0,
    )
    policy = RegretBrokeredMaterializedActionPolicy(MaterializedActionEvidenceLedger())
    decision = policy.select(
        MaterializedActionBrokerRequest(
            actions=(reference, synergistic, duplicate_route, distractor),
            evaluation_slots=2,
            slate_value=_SynergyValue(frozenset({"reference", "synergistic"})),
            slate_feasibility=_AlwaysFeasible(),
        )
    )

    assert decision.required_reference_action_sha256 == reference.action_sha256
    assert decision.selected_actions == tuple(
        sorted((reference, synergistic), key=lambda value: value.action_sha256)
    )
    assert len(
        {value.phenotype_identity_sha256 for value in decision.selected_actions}
    ) == 2


def test_multislot_reference_escrow_preserves_specialist_floor():
    context = _context(1, phase=SearchPhase.BASIN_ACQUISITION)
    references = tuple(
        _action(
            f"reference_{rank}",
            context=context,
            expert="numerical",
            operator="global",
            arity=0,
            rank=rank,
            reference=True,
        )
        for rank in range(1, 5)
    )
    residuals = tuple(
        _action(
            f"residual_{rank}",
            context=context,
            expert="semantic",
            operator="mutation",
            arity=1,
            rank=rank,
        )
        for rank in range(1, 5)
    )
    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    ).select(
        MaterializedActionBrokerRequest(
            actions=(*references, *residuals),
            evaluation_slots=4,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=3,
        )
    )

    required = {value.action_sha256 for value in references[:3]}
    assert set(decision.required_reference_action_sha256s) == required
    assert required.issubset(
        {value.action_sha256 for value in decision.selected_actions}
    )
    assert decision.reference_displaced_count == 0


def test_broker_honors_an_explicit_outcome_adaptive_directive():
    context = _context(1)
    actions = tuple(
        _action(
            f"adaptive_{rank}",
            context=context,
            expert="semantic",
            operator="mutation",
            arity=1,
            rank=rank,
        )
        for rank in range(1, 4)
    )
    selected = tuple(
        sorted(value.action_sha256 for value in actions[1:])
    )
    directive = AdaptiveActionAllocationDirective(
        policy_id="outcome_adaptive_test",
        policy_version=1,
        policy_definition_sha256=_sha("adaptive-policy"),
        residual_request_sha256=_sha("residual-request"),
        proposal_sha256s=(_sha("proposal"),),
        required_action_sha256s=selected,
        diagnostic_decision_sha256=_sha("diagnostic-decision"),
        continuation_decision_sha256s=(
            _sha("continuation-decision"),
        ),
        observed_outcome_sha256s=(
            _sha("adaptive-outcome"),
        ),
        observed_set_outcome_sha256s=(
            _sha("adaptive-set-outcome"),
        ),
        evidence=freeze_json(
            {
                "all_outcomes_observed_before_final_directive": True,
                "unobserved_candidate_outcomes_available": False,
            }
        ),
    )

    decision = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger()
    ).select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=2,
            slate_value=_SynergyValue(frozenset()),
            slate_feasibility=_AlwaysFeasible(),
            reference_escrow_slots=0,
            allocation_requirement=directive,
        )
    )

    assert tuple(
        value.action_sha256 for value in decision.selected_actions
    ) == selected
    assert decision.allocation_requirement is directive
    assert decision.to_record(include_allocation_evidence=True)[
        "allocation_requirement"
    ]["candidate_outcomes_observed"] is True
