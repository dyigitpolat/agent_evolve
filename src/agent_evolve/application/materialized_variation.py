"""Application adapters from deterministic variation policies to engine slots."""

from __future__ import annotations

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationPlan,
    MaterializedInvocation,
    OperatorKind,
)
from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.variation.disjoint_recombination import (
    DisjointPatchMaterialization,
    RecombinationBranch,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import CandidateDraft, SourceAttribution
from agent_evolve.ports.finite_action_selection import (
    FiniteActionDecision,
    FiniteActionSelectorKind,
    validate_finite_action_decision,
)
from agent_evolve.ports.id_factory import IdFactory


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the union.
            raise AssertionError("unsupported JSON-path segment")
    return "".join(parts)


def materialized_disjoint_invocation(
    *,
    plan: InvocationPlan,
    materialization: DisjointPatchMaterialization,
) -> MaterializedInvocation:
    """Bind one replay-verified disjoint union to its exact engine lineage."""

    if type(plan) is not InvocationPlan:
        raise TypeError("plan must be an exact InvocationPlan")
    InvocationPlan.__post_init__(plan)
    if plan.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION:
        raise ValueError("disjoint materialization requires a three-way plan")
    if type(materialization) is not DisjointPatchMaterialization:
        raise TypeError(
            "materialization must be an exact DisjointPatchMaterialization"
        )
    materialization.revalidate()
    if plan.common_ancestor is None:  # pragma: no cover - plan admission.
        raise ValueError("three-way plan requires a common ancestor")
    classification = materialization.classification
    if (
        classification.ancestor_candidate_id
        != plan.common_ancestor.candidate_id
        or classification.left_patch.target_candidate_id
        != plan.parents[0].candidate_id
        or classification.right_patch.target_candidate_id
        != plan.parents[1].candidate_id
    ):
        raise ValueError(
            "materialization endpoints differ from the invocation lineage"
        )
    if (
        classification.ancestor_hash
        != plan.common_ancestor.occurrence.configuration_hash
        or classification.left_patch.target_hash
        != plan.parents[0].occurrence.configuration_hash
        or classification.right_patch.target_hash
        != plan.parents[1].occurrence.configuration_hash
    ):
        raise ValueError(
            "materialization endpoint configurations differ from the "
            "invocation lineage"
        )

    source_attribution: list[SourceAttribution] = []
    intended_paths: list[str] = []
    for attribution in materialization.system_attribution:
        path = _path_text(attribution.path)
        intended_paths.append(path)
        if attribution.sources == (RecombinationBranch.LEFT,):
            source_attribution.append(SourceAttribution(path, "left"))
        elif attribution.sources == (RecombinationBranch.RIGHT,):
            source_attribution.append(SourceAttribution(path, "right"))
        elif attribution.sources == (
            RecombinationBranch.LEFT,
            RecombinationBranch.RIGHT,
        ):
            # An identical branch effect is preserved and receipt-bound, but a
            # single-source prose field cannot truthfully call it left or right.
            continue
        else:  # pragma: no cover - policy revalidation closes source sets.
            raise AssertionError("unsupported recombination attribution")
    configuration = thaw_json(materialization.configuration)
    if type(configuration) is not dict:
        raise TypeError("materialized candidate root must be an object")
    return MaterializedInvocation(
        plan=plan,
        draft=CandidateDraft(
            configuration=configuration,
            design_rationale=(
                "Engine-owned replay-verified union of two disjoint "
                "ancestor-relative typed patches."
            ),
            intended_changes=tuple(sorted(set(intended_paths))),
            source_attribution=tuple(source_attribution),
        ),
        candidate_id=materialization.union_patch.target_candidate_id,
        materialization_policy_id=materialization.policy_id,
        materialization_policy_version=materialization.policy_version,
        materialization_receipt_hash=materialization.receipt_sha256,
    )


def materialized_finite_action_decision(
    *,
    ids: IdFactory,
    parent: EvolutionCandidate,
    generation: int,
    label: str,
    authority: FiniteActionSetAuthority,
    decision: FiniteActionDecision,
    phase: str = "engine_finite_action",
) -> MaterializedInvocation:
    """Materialize an engine decision through the normal evolution lineage."""

    if not isinstance(ids, IdFactory):
        raise TypeError("ids must implement IdFactory")
    if type(parent) is not EvolutionCandidate:
        raise TypeError("parent must be an exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(parent)
    if type(authority) is not FiniteActionSetAuthority:
        raise TypeError("authority must be an exact FiniteActionSetAuthority")
    FiniteActionSetAuthority.__post_init__(authority)
    validate_finite_action_decision(authority, decision)
    if decision.selector_kind is not FiniteActionSelectorKind.ENGINE:
        raise ValueError("engine materialization requires an engine decision")
    if (
        authority.support.parent_candidate_id != parent.candidate_id
        or authority.support.parent_configuration_sha256
        != parent.occurrence.configuration_hash
        or authority.support.support_contract.parent_configuration
        != parent.configuration
    ):
        raise ValueError("finite action authority is bound to a different parent")
    row = authority.support.options[decision.selected_ordinal]
    candidate_id = ids.new_candidate_id()
    patch = derive_patch(
        parent.configuration,
        row.option.child_configuration,
        base_candidate_id=parent.candidate_id,
        target_candidate_id=candidate_id,
    )
    paths = tuple(sorted({_path_text(operation.path) for operation in patch.operations}))
    if paths != row.changed_paths:
        raise ValueError("finite action materialization changed its authorized paths")
    top_level = tuple(
        sorted(
            {
                operation.path.segments[0].value
                for operation in patch.operations
                if type(operation.path.segments[0]) is ObjectKey
            }
        )
    )
    if not top_level:
        raise ValueError("finite action decision materialized no object path")
    configuration = thaw_json(row.option.child_configuration)
    if type(configuration) is not dict:
        raise TypeError("finite action child must be an object")
    return MaterializedInvocation(
        plan=InvocationPlan(
            operator_kind=OperatorKind.TYPED_MUTATION,
            parents=(parent,),
            generation=generation,
            label=label,
            allowed_top_level=top_level,
            phase=phase,
        ),
        draft=CandidateDraft(
            configuration=configuration,
            design_rationale=(
                "Engine-owned prospective selection from the exact matched "
                "finite action support."
            ),
            intended_changes=paths,
            source_attribution=tuple(
                SourceAttribution(path, "mutation") for path in paths
            ),
        ),
        candidate_id=candidate_id,
        materialization_policy_id=decision.selector_policy_id,
        materialization_policy_version=decision.selector_policy_version,
        materialization_receipt_hash=decision.decision_sha256,
        materialized_finite_action_authority=authority,
        materialized_finite_action_decision=decision,
    )


__all__ = [
    "materialized_disjoint_invocation",
    "materialized_finite_action_decision",
]
