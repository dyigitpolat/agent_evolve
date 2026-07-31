"""Provider-free qualification for the generic V13 source/operator controller."""

from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application import contextual_campaign_planning as planning_module
from agent_evolve.application.contextual_search_controller import (
    ContextualSearchDelayedCredit,
    ContextualSearchLedger,
    ContextualSearchObservation,
    ContextualSearchQuery,
    PhaseAwareContextualSearchController,
    SearchArmKind,
    SearchPhase,
    audit_completed_contextual_search_ledger,
    slice_contextual_search_decision,
)
from agent_evolve.application.contextual_campaign_planning import (
    FiniteContractContextualJointCapabilityProjector,
    _empirical_count_capability,
    _project_archive_front_size,
)
from agent_evolve.application.contextual_delayed_credit import (
    observe_contextual_terminal_persistence,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.ports.contextual_search_allocation import (
    ContextualJointCountVector,
    ContextualLaneJointCountCapability,
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    required_source_evaluation_option_ids,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(
    ordinal: int,
    *,
    source: str,
    operator: str,
    gain: float,
    persisted: bool,
    descendant: bool,
) -> ContextualSearchObservation:
    return ContextualSearchObservation(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=1,
        source_id=source,
        operator_id=operator,
        option_identity_sha256=_sha(f"option-{ordinal}"),
        parent_context_sha256=_sha("parent-g1"),
        feasible=True,
        positive_marginal_utility=gain > 0.0,
        normalized_marginal_utility=gain,
        marginal_utility_share=gain,
        candidate_id=CandidateId(f"candidate_controller_{ordinal:02d}"),
        final_front_persisted=persisted,
        useful_descendant_observed=descendant,
    )


def _query(wave: int, *, composition: bool = False) -> ContextualSearchQuery:
    return ContextualSearchQuery(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=wave,
        total_portfolio_waves=3,
        real_evaluation_slots=4,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
        incumbent_source_id="model",
        incumbent_operator_id="atomic",
        archive_front_size=3,
        recent_normalized_archive_gains=(0.2,),
        composition_evidence_available=composition,
    )


def _allocation(decision, kind: str) -> dict[str, int]:
    values = (
        decision.source_allocations
        if kind == "source"
        else decision.operator_allocations
    )
    return {value.arm_id: value.target_slots for value in values}


def test_archive_front_size_projection_accepts_generic_runtime_record() -> None:
    assert (
        _project_archive_front_size(
            {
                "summary": {"front_size": 2},
                "front_candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            }
        )
        == 2
    )


def test_archive_front_size_projection_fails_closed_on_inconsistent_witnesses() -> None:
    try:
        _project_archive_front_size(
            {
                "front_size": 2,
                "summary": {"front_size": 3},
                "front_candidates": [{}, {}],
            }
        )
    except ValueError as error:
        assert "witnesses disagree" in str(error)
    else:  # pragma: no cover - must fail closed.
        raise AssertionError("inconsistent archive witnesses were accepted")


def test_initial_decision_preserves_one_exploration_slot() -> None:
    ledger = ContextualSearchLedger()
    query = _query(1)
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )
    decision = PhaseAwareContextualSearchController().decide(query, snapshot)

    assert decision.phase is SearchPhase.BASIN_ACQUISITION
    assert _allocation(decision, "source") == {"engine_global": 2, "model": 2}
    assert _allocation(decision, "operator") == {"atomic": 3, "composite": 1}
    assert sum(value.exploration_slot for value in decision.source_allocations) == 1
    assert decision.to_record()["terminal_information_bonus"] is None


def _joint_capability(
    slice_id: str,
    ordinal: int,
    *,
    source_counts: tuple[tuple[str, int], ...],
    operator_counts: tuple[tuple[str, int], ...],
) -> ContextualLaneJointCountCapability:
    vector = ContextualJointCountVector(
        source_target_counts=source_counts,
        operator_target_counts=operator_counts,
        feasibility_witness_option_identity_sha256s=tuple(
            sorted(
                (
                    _sha(f"joint-{ordinal}-a"),
                    _sha(f"joint-{ordinal}-b"),
                )
            )
        ),
    )
    return ContextualLaneJointCountCapability(
        slice_id=slice_id,
        finite_contract_identity_sha256=_sha(f"contract-{ordinal}"),
        structural_constraint_sha256=_sha("joint-constraint"),
        evaluation_slots=2,
        source_arm_ids=("engine_global", "model"),
        operator_arm_ids=("atomic", "composite"),
        feasible_vectors=(vector,),
    )


def test_prospective_joint_capability_projects_before_exact_lane_slicing() -> None:
    capabilities = (
        _joint_capability(
            "elite",
            1,
            source_counts=(("engine_global", 0), ("model", 2)),
            operator_counts=(("atomic", 2), ("composite", 0)),
        ),
        _joint_capability(
            "explorer",
            2,
            source_counts=(("engine_global", 1), ("model", 1)),
            operator_counts=(("atomic", 1), ("composite", 1)),
        ),
    )
    query = ContextualSearchQuery(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=3,
        total_portfolio_waves=3,
        real_evaluation_slots=4,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
        incumbent_source_id="model",
        incumbent_operator_id="atomic",
        archive_front_size=3,
        joint_count_capabilities=capabilities,
    )
    ledger = ContextualSearchLedger()
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )
    decision = PhaseAwareContextualSearchController().decide(query, snapshot)

    # The independent terminal prior requests 2 global / 2 model slots.  The
    # exact lane product admits only 1 / 3, so projection occurs before a
    # provider-visible contract exists.
    assert _allocation(decision, "source") == {"engine_global": 1, "model": 3}
    assert _allocation(decision, "operator") == {"atomic": 3, "composite": 1}
    assert any(
        value.prospective_joint_capability_projected
        for value in decision.source_allocations
    )
    decision_record = decision.to_record()
    assert len(decision_record["query"]["joint_count_capabilities"]) == 2
    assert len(decision_record["joint_capability_selection"]) == 2
    assert decision_record["snapshot"]["cutoff_wave_index_exclusive"] == 3
    stage = slice_contextual_search_decision(
        decision,
        slice_ids=("elite", "explorer"),
        evaluation_slots=(2, 2),
    )
    assert [dict(value.source_target_counts) for value in stage.slices] == [
        {"engine_global": 0, "model": 2},
        {"engine_global": 1, "model": 1},
    ]
    assert [dict(value.operator_target_counts) for value in stage.slices] == [
        {"atomic": 2, "composite": 0},
        {"atomic": 1, "composite": 1},
    ]
    assert [
        value.feasibility_witness_option_identity_sha256s
        for value in stage.slices
    ] == [
        capabilities[0].feasible_vectors[0].feasibility_witness_option_identity_sha256s,
        capabilities[1].feasible_vectors[0].feasibility_witness_option_identity_sha256s,
    ]
    assert [
        value.to_contract(
            campaign_generation=5
        ).feasibility_witness_option_identity_sha256s
        for value in stage.slices
    ] == [
        capabilities[0].feasible_vectors[0].feasibility_witness_option_identity_sha256s,
        capabilities[1].feasible_vectors[0].feasibility_witness_option_identity_sha256s,
    ]


def test_joint_capability_reassigns_an_unrealizable_exploration_arm() -> None:
    source_ids = ("engine_global", "model")
    operator_ids = ("atomic", "composite", "global")
    source_preserving = ContextualJointCountVector(
        source_target_counts=(("engine_global", 1), ("model", 3)),
        operator_target_counts=(("atomic", 3), ("composite", 0), ("global", 1)),
        feasibility_witness_option_identity_sha256s=tuple(
            sorted(_sha(f"reassigned-a-{index}") for index in range(4))
        ),
    )
    operator_preserving = ContextualJointCountVector(
        source_target_counts=(("engine_global", 0), ("model", 4)),
        operator_target_counts=(("atomic", 3), ("composite", 1), ("global", 0)),
        feasibility_witness_option_identity_sha256s=tuple(
            sorted(_sha(f"reassigned-b-{index}") for index in range(4))
        ),
    )
    capability = ContextualLaneJointCountCapability(
        slice_id="only_lane",
        finite_contract_identity_sha256=_sha("reassigned-contract"),
        structural_constraint_sha256=_sha("reassigned-constraint"),
        evaluation_slots=4,
        source_arm_ids=source_ids,
        operator_arm_ids=operator_ids,
        feasible_vectors=tuple(
            sorted(
                (source_preserving, operator_preserving),
                key=lambda value: value.vector_sha256,
            )
        ),
    )
    query = ContextualSearchQuery(
        campaign_scope_sha256=_sha("reassigned-campaign"),
        wave_index=1,
        total_portfolio_waves=3,
        real_evaluation_slots=4,
        available_source_ids=source_ids,
        available_operator_ids=operator_ids,
        incumbent_source_id="model",
        incumbent_operator_id="atomic",
        archive_front_size=2,
        joint_count_capabilities=(capability,),
    )
    ledger = ContextualSearchLedger()
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )

    decision = PhaseAwareContextualSearchController().decide(query, snapshot)

    assert _allocation(decision, "source") == {"engine_global": 1, "model": 3}
    assert _allocation(decision, "operator") == {
        "atomic": 3,
        "composite": 0,
        "global": 1,
    }
    assert next(
        value.arm_id for value in decision.source_allocations if value.exploration_slot
    ) == "engine_global"
    assert next(
        value.arm_id
        for value in decision.operator_allocations
        if value.exploration_slot
    ) == "global"
    projected = {
        value.arm_id
        for value in decision.operator_allocations
        if value.prospective_joint_exploration_projected
    }
    assert projected == {"composite", "global"}
    decision.__post_init__()


class _JointProjectionContext:
    def __init__(
        self,
        contract: FiniteVariationContract,
        *,
        future_recombination_width: int | None = None,
    ) -> None:
        if future_recombination_width is None:
            self.prepared = object()
            self.stage_request = object()
        else:
            future = type(
                "FutureStep",
                (),
                {
                    "source_portfolio_generation": 1,
                    "offspring_per_parent": future_recombination_width,
                },
            )()
            self.prepared = type(
                "Prepared",
                (),
                {"schedule": type("Schedule", (), {"steps": (future,)})()},
            )()
            current = type("CurrentStep", (), {"generation": 1})()
            self.stage_request = type(
                "StageRequest",
                (),
                {"step": current},
            )()
        self.parent_lane = type("Lane", (), {"lane_id": "elite"})()
        self.variation = type("Variation", (), {"contract": contract})()

    def __post_init__(self) -> None:
        return None


def test_finite_contract_joint_capability_rejects_impossible_source_mix() -> None:
    parent = freeze_json({"x": 0})
    parent_sha256 = typed_json_sha256(parent)
    options = []
    for index in range(4):
        options.append(
            FiniteVariationOption(
                option_id=f"global.{index}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": index + 1}),
                family="global_restart",
                description=f"Global restart {index}.",
                metadata=(
                    ("evaluation_source", "engine_global"),
                    ("evaluation_source_minimum", "1"),
                ),
            )
        )
    for index, family in enumerate(("local_a", "local_b", "local_c"), start=5):
        options.append(
            FiniteVariationOption(
                option_id=f"primary.{index}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": index}),
                family=family,
                description=f"Primary action {index}.",
            )
        )
    options.append(
        FiniteVariationOption(
            option_id="primary.composite",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"x": 9}),
            family="composite_r2",
            description="Primary composite action.",
            metadata=(("composition_radius", "2"),),
        )
    )
    contract = FiniteVariationContract(
        catalog_id="joint_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("joint-fixture"),
        parent_configuration=parent,
        options=tuple(options),
    )
    capability = FiniteContractContextualJointCapabilityProjector(
        min_distinct_families=3,
        operator_exposure_bounds=(("atomic", 1, 4),),
    ).project(
        _JointProjectionContext(contract),
        evaluation_slots=4,
        source_arm_ids=("engine_global", "primary"),
        operator_arm_ids=("atomic", "composite"),
    )

    source_rows = [
        dict(value.source_target_counts) for value in capability.feasible_vectors
    ]
    assert {"engine_global": 2, "primary": 2} in source_rows
    assert {"engine_global": 3, "primary": 1} not in source_rows
    assert all(
        dict(value.operator_target_counts)["atomic"] >= 1
        for value in capability.feasible_vectors
    )
    assert all(
        record["objective_values_consulted"] is False
        and record["workload_identifiers_consulted"] is False
        for record in (value.to_record() for value in capability.feasible_vectors)
    )


def test_joint_capability_uses_semantic_source_floor_not_fixed_representative() -> None:
    parent = freeze_json({"a": 0, "b": 0, "d": 0})
    parent_sha256 = typed_json_sha256(parent)
    options = (
        FiniteVariationOption(
            option_id="global.clashing",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"a": 1, "b": 0, "d": 0}),
            family="global",
            description="Global source action that clashes with primary A.",
            metadata=(
                ("evaluation_source", "global_restart"),
                ("evaluation_source_minimum", "1"),
            ),
        ),
        FiniteVariationOption(
            option_id="global.compatible",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"a": 0, "b": 0, "d": 1}),
            family="global",
            description="Compatible action from the same semantic source.",
            metadata=(
                ("evaluation_source", "global_restart"),
                ("evaluation_source_minimum", "1"),
            ),
        ),
        FiniteVariationOption(
            option_id="primary.a",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"a": 2, "b": 0, "d": 0}),
            family="primary_a",
            description="Primary A action.",
        ),
        FiniteVariationOption(
            option_id="primary.b",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"a": 0, "b": 2, "d": 0}),
            family="primary_b",
            description="Primary B action.",
        ),
    )
    contract = None
    for salt in range(256):
        candidate = FiniteVariationContract(
            catalog_id="semantic_source_floor_fixture",
            catalog_version=1,
            catalog_definition_sha256=_sha(f"semantic-floor-{salt}"),
            parent_configuration=parent,
            options=options,
        )
        if required_source_evaluation_option_ids(candidate) == ("global.clashing",):
            contract = candidate
            break
    assert contract is not None

    capability = FiniteContractContextualJointCapabilityProjector(
        min_distinct_families=3,
        require_pairwise_disjoint_parent_patches=True,
    ).project(
        _JointProjectionContext(contract),
        evaluation_slots=3,
        source_arm_ids=("global_restart", "primary"),
        operator_arm_ids=("atomic", "composite"),
    )
    vector = next(
        value
        for value in capability.feasible_vectors
        if dict(value.source_target_counts) == {"global_restart": 1, "primary": 2}
        and dict(value.operator_target_counts) == {"atomic": 3, "composite": 0}
    )
    compatible_identity = contract.resolve("global.compatible").identity_sha256
    clashing_identity = contract.resolve("global.clashing").identity_sha256
    assert compatible_identity in vector.feasibility_witness_option_identity_sha256s
    assert clashing_identity not in vector.feasibility_witness_option_identity_sha256s


def test_pairwise_joint_capability_collapses_exact_feasibility_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value multiplicity at one locus must not inflate exact witness search."""

    path_count = 12
    values_per_path = 8
    parent_payload = {f"p{index:02d}": 0 for index in range(path_count)}
    parent = freeze_json(parent_payload)
    parent_sha256 = typed_json_sha256(parent)
    options = []
    for path_index in range(path_count):
        for value in range(1, values_per_path + 1):
            child = dict(parent_payload)
            child[f"p{path_index:02d}"] = value
            options.append(
                FiniteVariationOption(
                    option_id=f"dense.p{path_index:02d}.v{value:02d}",
                    parent_configuration_sha256=parent_sha256,
                    child_configuration=freeze_json(child),
                    family="dense_atomic",
                    description="Dense scalar-grid alternative at one locus.",
                )
            )
    contract = FiniteVariationContract(
        catalog_id="dense_pairwise_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("dense-pairwise-fixture"),
        parent_configuration=parent,
        options=tuple(options),
    )
    original = planning_module.pairwise_disjoint_parent_patch_pairs
    projected_domain_sizes: list[int] = []

    def observed_pairs(active_contract, option_ids):
        projected_domain_sizes.append(len(option_ids))
        return original(active_contract, option_ids)

    monkeypatch.setattr(
        planning_module,
        "pairwise_disjoint_parent_patch_pairs",
        observed_pairs,
    )
    capability = FiniteContractContextualJointCapabilityProjector(
        require_pairwise_disjoint_parent_patches=True,
        minimum_single_path_interventions=4,
    ).project(
        _JointProjectionContext(contract),
        evaluation_slots=4,
        source_arm_ids=("primary",),
        operator_arm_ids=("atomic",),
    )

    assert projected_domain_sizes == [path_count]
    assert len(capability.feasible_vectors) == 1
    witness_identities = set(
        capability.feasible_vectors[0].feasibility_witness_option_identity_sha256s
    )
    witness_ids = {
        option.option_id
        for option in contract.options
        if option.identity_sha256 in witness_identities
    }
    assert len(witness_ids) == 4
    assert len(pairwise_disjoint_parent_patch_pairs(contract, tuple(witness_ids))) == 6


def test_joint_capability_reserves_true_single_path_assay_independent_of_operator() -> None:
    parent = freeze_json({"x": 0, "y": 0, "z": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="intervention_axis_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("intervention-axis-fixture"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="atomic.joint_a",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 1, "y": 1, "z": 0}),
                family="same_declared_operator",
                description="Declared atomic proposal with two changed paths.",
            ),
            FiniteVariationOption(
                option_id="atomic.joint_b",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 2, "y": 2, "z": 0}),
                family="same_declared_operator",
                description="Second declared atomic proposal with two paths.",
            ),
            FiniteVariationOption(
                option_id="atomic.single",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 0, "y": 0, "z": 1}),
                family="same_declared_operator",
                description="Declared atomic proposal with one changed path.",
            ),
        ),
    )
    capability = FiniteContractContextualJointCapabilityProjector(
        minimum_single_path_interventions=1,
    ).project(
        _JointProjectionContext(contract),
        evaluation_slots=2,
        source_arm_ids=("primary",),
        operator_arm_ids=("atomic",),
    )

    assert capability.minimum_single_path_interventions == 1
    assert len(capability.feasible_vectors) == 1
    assert contract.resolve("atomic.single").identity_sha256 in (
        capability.feasible_vectors[0].feasibility_witness_option_identity_sha256s
    )
    assert capability.to_record()["intervention_axis"] == (
        "exact_parent_relative_changed_json_path_count"
    )


def test_joint_capability_reserves_registered_future_recombination_opportunities() -> None:
    parent = freeze_json({"a": 0, "b": 0, "c": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="offspring_opportunity_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("offspring-opportunity-fixture"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="action.a1",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 1, "b": 0, "c": 0}),
                family="same_family",
                description="First action on A.",
            ),
            FiniteVariationOption(
                option_id="action.a2",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 2, "b": 0, "c": 0}),
                family="same_family",
                description="Second action on A.",
            ),
            FiniteVariationOption(
                option_id="action.b",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 0, "b": 1, "c": 0}),
                family="same_family",
                description="Action on B.",
            ),
            FiniteVariationOption(
                option_id="action.c",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"a": 0, "b": 0, "c": 1}),
                family="same_family",
                description="Action on C.",
            ),
        ),
    )
    capability = FiniteContractContextualJointCapabilityProjector(
        protect_future_recombination_opportunities=True,
    ).project(
        _JointProjectionContext(contract, future_recombination_width=2),
        evaluation_slots=3,
        source_arm_ids=("primary",),
        operator_arm_ids=("atomic",),
    )

    assert capability.minimum_disjoint_parent_patch_pairs == 2
    identity_to_option = {
        option.identity_sha256: option.option_id for option in contract.options
    }
    for vector in capability.feasible_vectors:
        witness = tuple(
            identity_to_option[value]
            for value in vector.feasibility_witness_option_identity_sha256s
        )
        assert len(pairwise_disjoint_parent_patch_pairs(contract, witness)) >= 2
    assert capability.to_record()["offspring_opportunity_axis"] == (
        "pairwise_disjoint_parent_relative_patch_pairs"
    )


def test_joint_capability_continues_after_unrepairable_base_signature_set() -> None:
    parent = freeze_json({"x": 0, "y": 0, "z": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="offspring_opportunity_signature_recourse_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("offspring-signature-recourse"),
        parent_configuration=parent,
        options=(
            FiniteVariationOption(
                option_id="family_a.x",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 1, "y": 0, "z": 0}),
                family="family_a",
                description="First action on X.",
            ),
            FiniteVariationOption(
                option_id="family_b.x",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 2, "y": 0, "z": 0}),
                family="family_b",
                description="Second family action on X.",
            ),
            FiniteVariationOption(
                option_id="family_c.xy",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 3, "y": 1, "z": 0}),
                family="family_c",
                description="Joint action overlapping both early signatures.",
            ),
            FiniteVariationOption(
                option_id="family_d.z",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json({"x": 0, "y": 0, "z": 1}),
                family="family_d",
                description="Later signature that restores two pair opportunities.",
            ),
        ),
    )
    capability = FiniteContractContextualJointCapabilityProjector(
        min_distinct_families=3,
        minimum_single_path_interventions=1,
        protect_future_recombination_opportunities=True,
    ).project(
        _JointProjectionContext(contract, future_recombination_width=2),
        evaluation_slots=3,
        source_arm_ids=("primary",),
        operator_arm_ids=("atomic",),
    )

    assert len(capability.feasible_vectors) == 1
    identity_to_option = {
        option.identity_sha256: option.option_id for option in contract.options
    }
    witness = tuple(
        identity_to_option[value]
        for value in capability.feasible_vectors[0].feasibility_witness_option_identity_sha256s
    )
    assert "family_d.z" in witness
    assert len(pairwise_disjoint_parent_patch_pairs(contract, witness)) >= 2


def test_sealed_outcomes_reallocate_and_terminal_removes_information_slot() -> None:
    values = (
        _observation(
            1,
            source="model",
            operator="atomic",
            gain=0.4,
            persisted=True,
            descendant=True,
        ),
        _observation(
            2,
            source="model",
            operator="atomic",
            gain=0.3,
            persisted=True,
            descendant=False,
        ),
        _observation(
            3,
            source="model",
            operator="composite",
            gain=0.0,
            persisted=False,
            descendant=False,
        ),
        _observation(
            4,
            source="engine_global",
            operator="composite",
            gain=0.0,
            persisted=False,
            descendant=False,
        ),
    )
    ledger = ContextualSearchLedger()
    ledger.append_batch(
        tuple(sorted(values, key=lambda value: value.observation_sha256))
    )

    expansion_query = _query(2)
    expansion_snapshot = ledger.snapshot(
        campaign_scope_sha256=expansion_query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=expansion_query.wave_index,
        available_source_ids=expansion_query.available_source_ids,
        available_operator_ids=expansion_query.available_operator_ids,
    )
    expansion = PhaseAwareContextualSearchController().decide(
        expansion_query,
        expansion_snapshot,
    )
    assert expansion.phase is SearchPhase.BASIN_EXPANSION
    # Exploitation follows observed archive return; the zero-return source and
    # operator retain exactly one nonterminal uncertainty slot, not a second
    # pseudo-prior allocation.
    assert _allocation(expansion, "source") == {"engine_global": 1, "model": 3}
    assert _allocation(expansion, "operator") == {"atomic": 3, "composite": 1}

    terminal_query = _query(3)
    terminal_snapshot = ledger.snapshot(
        campaign_scope_sha256=terminal_query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=terminal_query.wave_index,
        available_source_ids=terminal_query.available_source_ids,
        available_operator_ids=terminal_query.available_operator_ids,
    )
    terminal = PhaseAwareContextualSearchController().decide(
        terminal_query,
        terminal_snapshot,
    )
    assert terminal.phase is SearchPhase.TERMINAL_CONVERSION
    assert _allocation(terminal, "source") == {"engine_global": 0, "model": 4}
    assert _allocation(terminal, "operator") == {"atomic": 4, "composite": 0}
    assert not any(value.exploration_slot for value in terminal.source_allocations)
    assert terminal.to_record()["terminal_information_bonus"] == 0.0


def test_composition_phase_is_explicit_and_auditable() -> None:
    value = _observation(
        1,
        source="model",
        operator="atomic",
        gain=0.2,
        persisted=True,
        descendant=True,
    )
    ledger = ContextualSearchLedger()
    ledger.append_batch((value,))
    query = _query(2, composition=True)
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )
    decision = PhaseAwareContextualSearchController().decide(query, snapshot)
    assert decision.phase is SearchPhase.COMPOSITION
    assert decision.snapshot.observation_sha256s == (value.observation_sha256,)


def test_delayed_descendant_credit_obeys_its_availability_cutoff() -> None:
    value = _observation(
        1,
        source="model",
        operator="atomic",
        gain=0.0,
        persisted=False,
        descendant=False,
    )
    # Model the fact as unavailable at portfolio close; a later recombination
    # stage is the only authority allowed to publish it.
    value = ContextualSearchObservation(
        campaign_scope_sha256=value.campaign_scope_sha256,
        wave_index=value.wave_index,
        source_id=value.source_id,
        operator_id=value.operator_id,
        option_identity_sha256=value.option_identity_sha256,
        parent_context_sha256=value.parent_context_sha256,
        feasible=value.feasible,
        positive_marginal_utility=value.positive_marginal_utility,
        normalized_marginal_utility=value.normalized_marginal_utility,
        marginal_utility_share=value.marginal_utility_share,
        candidate_id=value.candidate_id,
        final_front_persisted=None,
        useful_descendant_observed=None,
    )
    ledger = ContextualSearchLedger()
    ledger.append_batch((value,))
    credit = ContextualSearchDelayedCredit(
        campaign_scope_sha256=value.campaign_scope_sha256,
        source_observation_sha256=value.observation_sha256,
        available_at_wave_index=3,
        useful_descendant_observed=True,
    )
    ledger.append_delayed_credit_batch((credit,))

    before = ledger.snapshot(
        campaign_scope_sha256=value.campaign_scope_sha256,
        cutoff_wave_index_exclusive=2,
        available_source_ids=("model",),
        available_operator_ids=("atomic",),
    )
    after = ledger.snapshot(
        campaign_scope_sha256=value.campaign_scope_sha256,
        cutoff_wave_index_exclusive=3,
        available_source_ids=("model",),
        available_operator_ids=("atomic",),
    )

    assert before.source_posteriors[0].descendant_observation_count == 0
    assert before.delayed_credit_sha256s == ()
    assert after.source_posteriors[0].descendant_observation_count == 1
    assert after.source_posteriors[0].descendant_positive_count == 1
    assert after.delayed_credit_sha256s == (credit.credit_sha256,)


def test_allocation_recourse_is_prior_only_arm_realizability_evidence() -> None:
    ledger = ContextualSearchLedger()
    realization = ContextualPortfolioAllocationRealization(
        campaign_scope_sha256=_sha("controller-campaign"),
        query_sha256=_sha("allocation-query"),
        decision_sha256=_sha("allocation-decision"),
        contract_sha256=_sha("allocation-contract"),
        controller_wave_index=1,
        slice_id="elite",
        requested_source_target_counts=(
            ("engine_global", 1),
            ("model", 3),
        ),
        requested_operator_target_counts=(
            ("atomic", 1),
            ("composite", 3),
        ),
        realized_source_target_counts=(
            ("engine_global", 1),
            ("model", 3),
        ),
        realized_operator_target_counts=(
            ("atomic", 2),
            ("composite", 2),
        ),
    )
    ledger.append_allocation_realization_batch((realization,))

    before = ledger.snapshot(
        campaign_scope_sha256=realization.campaign_scope_sha256,
        cutoff_wave_index_exclusive=1,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
    )
    after = ledger.snapshot(
        campaign_scope_sha256=realization.campaign_scope_sha256,
        cutoff_wave_index_exclusive=2,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
    )

    assert before.allocation_realization_sha256s == ()
    assert after.allocation_realization_sha256s == (realization.realization_sha256,)
    model = next(value for value in after.source_posteriors if value.arm_id == "model")
    composite = next(
        value for value in after.operator_posteriors if value.arm_id == "composite"
    )
    assert (
        model.allocation_requested_slot_count,
        model.allocation_realized_overlap_count,
        model.allocation_realizability_probability,
    ) == (3, 3, 0.8)
    assert (
        composite.allocation_requested_slot_count,
        composite.allocation_realized_overlap_count,
        composite.allocation_realizability_probability,
    ) == (3, 2, 0.6)


def test_zero_requested_arm_does_not_create_phantom_recourse_evidence() -> None:
    ledger = ContextualSearchLedger()
    realization = ContextualPortfolioAllocationRealization(
        campaign_scope_sha256=_sha("controller-campaign"),
        query_sha256=_sha("zero-allocation-query"),
        decision_sha256=_sha("zero-allocation-decision"),
        contract_sha256=_sha("zero-allocation-contract"),
        controller_wave_index=1,
        slice_id="elite",
        requested_source_target_counts=(
            ("engine_global", 0),
            ("model", 4),
        ),
        requested_operator_target_counts=(
            ("atomic", 0),
            ("composite", 4),
        ),
        realized_source_target_counts=(
            ("engine_global", 1),
            ("model", 3),
        ),
        realized_operator_target_counts=(
            ("atomic", 1),
            ("composite", 3),
        ),
    )
    ledger.append_allocation_realization_batch((realization,))
    snapshot = ledger.snapshot(
        campaign_scope_sha256=realization.campaign_scope_sha256,
        cutoff_wave_index_exclusive=2,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
    )

    engine = next(
        value for value in snapshot.source_posteriors if value.arm_id == "engine_global"
    )
    atomic = next(
        value for value in snapshot.operator_posteriors if value.arm_id == "atomic"
    )
    assert engine.allocation_requested_slot_count == 0
    assert engine.allocation_projection_count == 0
    assert atomic.allocation_requested_slot_count == 0
    assert atomic.allocation_projection_count == 0


def test_prior_realized_count_witness_projects_repeated_impossible_request() -> None:
    realizations = tuple(
        sorted(
            (
                ContextualPortfolioAllocationRealization(
                    campaign_scope_sha256=_sha("controller-campaign"),
                    query_sha256=_sha(f"capability-query-{lane}"),
                    decision_sha256=_sha(f"capability-decision-{lane}"),
                    contract_sha256=_sha(f"capability-contract-{lane}"),
                    controller_wave_index=1,
                    slice_id=lane,
                    requested_source_target_counts=(
                        ("engine_global", 1),
                        ("model", 3),
                    ),
                    requested_operator_target_counts=(
                        ("atomic", 2),
                        ("composite", 2),
                    ),
                    realized_source_target_counts=(
                        ("engine_global", 1),
                        ("model", 3),
                    ),
                    realized_operator_target_counts=(
                        ("atomic", 3 if lane == "elite" else 4),
                        ("composite", 1 if lane == "elite" else 0),
                    ),
                )
                for lane in ("elite", "explorer")
            ),
            key=lambda value: value.realization_sha256,
        )
    )
    ledger = ContextualSearchLedger()
    ledger.append_allocation_realization_batch(realizations)
    capability = _empirical_count_capability(
        realizations,
        kind="operator",
        current_wave_index=2,
        evaluation_slots=8,
        arm_ids=("atomic", "composite"),
    )
    assert capability is not None
    assert capability.feasible_count_vectors == ((("atomic", 7), ("composite", 1)),)
    query = ContextualSearchQuery(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=2,
        total_portfolio_waves=3,
        real_evaluation_slots=8,
        available_source_ids=("engine_global", "model"),
        available_operator_ids=("atomic", "composite"),
        incumbent_source_id="model",
        incumbent_operator_id="atomic",
        archive_front_size=3,
        operator_count_capability=capability,
    )
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )
    decision = PhaseAwareContextualSearchController().decide(query, snapshot)

    assert _allocation(decision, "operator") == {"atomic": 7, "composite": 1}
    assert {
        value.arm_id: value.unconstrained_target_slots
        for value in decision.operator_allocations
    } == {"atomic": 6, "composite": 2}
    assert all(
        value.empirical_capability_projected for value in decision.operator_allocations
    )


def test_terminal_persistence_remains_a_separate_post_campaign_label() -> None:
    first = _observation(
        1,
        source="model",
        operator="atomic",
        gain=0.2,
        persisted=False,
        descendant=False,
    )
    second = _observation(
        2,
        source="engine_global",
        operator="composite",
        gain=0.0,
        persisted=False,
        descendant=False,
    )
    observations = tuple(
        sorted(
            (
                ContextualSearchObservation(
                    campaign_scope_sha256=value.campaign_scope_sha256,
                    wave_index=value.wave_index,
                    source_id=value.source_id,
                    operator_id=value.operator_id,
                    option_identity_sha256=value.option_identity_sha256,
                    parent_context_sha256=value.parent_context_sha256,
                    feasible=value.feasible,
                    positive_marginal_utility=value.positive_marginal_utility,
                    normalized_marginal_utility=(value.normalized_marginal_utility),
                    marginal_utility_share=value.marginal_utility_share,
                    candidate_id=value.candidate_id,
                )
                for value in (first, second)
            ),
            key=lambda value: value.observation_sha256,
        )
    )
    terminal = first.candidate_id
    assert terminal is not None
    batch = observe_contextual_terminal_persistence(
        campaign_scope_sha256=first.campaign_scope_sha256,
        available_at_wave_index=4,
        finalization_request_sha256=_sha("finalization-request"),
        observations=observations,
        terminal_front_candidate_ids=(terminal,),
    )

    source_by_hash = {value.observation_sha256: value for value in observations}
    for credit in batch.credits:
        source = source_by_hash[credit.source_observation_sha256]
        assert credit.stage_front_persisted is None
        assert credit.useful_descendant_observed is None
        assert credit.final_front_persisted is (source.candidate_id == terminal)


def test_completed_ledger_audit_requires_both_delayed_horizons() -> None:
    first = ContextualSearchObservation(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=1,
        source_id="model",
        operator_id="atomic",
        option_identity_sha256=_sha("audit-option-1"),
        parent_context_sha256=_sha("audit-parent"),
        feasible=True,
        positive_marginal_utility=True,
        normalized_marginal_utility=0.2,
        marginal_utility_share=1.0,
        candidate_id=CandidateId("candidate_audit_01"),
    )
    second = ContextualSearchObservation(
        campaign_scope_sha256=_sha("controller-campaign"),
        wave_index=1,
        source_id="engine_global",
        operator_id="composite",
        option_identity_sha256=_sha("audit-option-2"),
        parent_context_sha256=_sha("audit-parent"),
        feasible=True,
        positive_marginal_utility=False,
        normalized_marginal_utility=0.0,
        marginal_utility_share=0.0,
        candidate_id=CandidateId("candidate_audit_02"),
    )
    observations = tuple(
        sorted((first, second), key=lambda value: value.observation_sha256)
    )
    realization = ContextualPortfolioAllocationRealization(
        campaign_scope_sha256=first.campaign_scope_sha256,
        query_sha256=_sha("audit-query"),
        decision_sha256=_sha("audit-decision"),
        contract_sha256=_sha("audit-contract"),
        controller_wave_index=1,
        slice_id="all",
        requested_source_target_counts=(
            ("engine_global", 1),
            ("model", 1),
        ),
        requested_operator_target_counts=(
            ("atomic", 1),
            ("composite", 1),
        ),
        realized_source_target_counts=(
            ("engine_global", 1),
            ("model", 1),
        ),
        realized_operator_target_counts=(
            ("atomic", 1),
            ("composite", 1),
        ),
    )
    stage_credits = tuple(
        sorted(
            (
                ContextualSearchDelayedCredit(
                    campaign_scope_sha256=value.campaign_scope_sha256,
                    source_observation_sha256=value.observation_sha256,
                    available_at_wave_index=2,
                    stage_front_persisted=value is first,
                    useful_descendant_observed=value is first,
                )
                for value in observations
            ),
            key=lambda value: value.credit_sha256,
        )
    )
    terminal = first.candidate_id
    assert terminal is not None
    terminal_batch = observe_contextual_terminal_persistence(
        campaign_scope_sha256=first.campaign_scope_sha256,
        available_at_wave_index=3,
        finalization_request_sha256=_sha("audit-finalization"),
        observations=observations,
        terminal_front_candidate_ids=(terminal,),
    )
    ledger = ContextualSearchLedger()
    ledger.append_batch(observations)
    ledger.append_allocation_realization_batch((realization,))
    ledger.append_delayed_credit_batch(stage_credits)
    ledger.append_delayed_credit_batch(terminal_batch.credits)
    audit = audit_completed_contextual_search_ledger(
        ledger,
        campaign_scope_sha256=first.campaign_scope_sha256,
        expected_wave_count=1,
        expected_post_recombination_wave_indices=(1,),
        expected_observation_count=2,
        expected_allocation_realization_count=1,
    )

    assert audit.healthy is True
    assert audit.delayed_credit_count == 4
    assert audit.observation_wave_indices == (1,)
    assert audit.allocation_wave_indices == (1,)
    assert audit.descendant_credit_wave_indices == (1,)


def test_stage_decision_slices_exactly_across_concurrent_parent_requests() -> None:
    query = _query(1)
    ledger = ContextualSearchLedger()
    snapshot = ledger.snapshot(
        campaign_scope_sha256=query.campaign_scope_sha256,
        cutoff_wave_index_exclusive=query.wave_index,
        available_source_ids=query.available_source_ids,
        available_operator_ids=query.available_operator_ids,
    )
    decision = PhaseAwareContextualSearchController().decide(query, snapshot)
    allocation = slice_contextual_search_decision(
        decision,
        slice_ids=("elite", "explorer"),
        evaluation_slots=(2, 2),
    )

    assert allocation == slice_contextual_search_decision(
        decision,
        slice_ids=("elite", "explorer"),
        evaluation_slots=(2, 2),
    )
    assert [value.evaluation_slots for value in allocation.slices] == [2, 2]
    assert sum(
        value.target_count(SearchArmKind.SOURCE, "engine_global")
        for value in allocation.slices
    ) == next(
        value.target_slots
        for value in decision.source_allocations
        if value.arm_id == "engine_global"
    )
    assert sum(
        value.target_count(SearchArmKind.OPERATOR, "composite")
        for value in allocation.slices
    ) == next(
        value.target_slots
        for value in decision.operator_allocations
        if value.arm_id == "composite"
    )
    assert allocation.to_record()["decision"]["decision_sha256"] == (
        decision.decision_sha256
    )
