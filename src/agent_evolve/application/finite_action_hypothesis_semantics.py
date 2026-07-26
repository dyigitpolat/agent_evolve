"""Portable conservative semantics for reflected finite-action hypotheses.

Free-text triggers and mechanisms are never parsed.  This compiler deliberately
uses authenticated direct evidence when available to compile a replayable
parent-local transition claim.  Legacy or foreign evidence retains the older
all-parent interpretation so existing non-identifiable workflows remain
explicitly backward compatible.  A workload may inject a narrower semantic
compiler and matcher, but this implementation provides one shared default for
BOiLS, Heat, Timeloop, and other finite catalogs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent_evolve.application.finite_action_transition import (
    EmpiricalFiniteActionTransition,
    empirical_finite_action_transitions_for_insight,
)
from agent_evolve.application.campaign_learning_runtime import (
    CampaignInsightSemanticCompiler,
    CompiledCampaignInsightSemantics,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    GlobalHypothesisAuditRequest,
    HypothesisEvidenceMatchReceipt,
    InterventionIdentifiability,
    InterventionMatch,
    TriggerMatch,
    TypedEvidencePredicate,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    ReflectionInsightContract,
)
from agent_evolve.application.insight_memory import InsightEvidenceLineage


PORTABLE_FINITE_ACTION_MATCHER_ID = "portable_finite_action_hypothesis_matcher"
PORTABLE_FINITE_ACTION_MATCHER_VERSION = 2
PORTABLE_FINITE_ACTION_MATCHER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portable-finite-action-hypothesis-matcher:v2:"
    b"authenticated-parent-local-transition-when-available:"
    b"stable-action-id-family-path-and-compiler:typed-old-new-values:"
    b"legacy-all-parent-schema-compatible"
).hexdigest()
PORTABLE_FINITE_ACTION_COMPILER_ID = "portable_finite_action_insight_compiler"
PORTABLE_FINITE_ACTION_COMPILER_VERSION = 2
PORTABLE_FINITE_ACTION_COMPILER_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portable-finite-action-insight-compiler:v2:"
    b"no-prose-parsing:authenticated-direct-transition-first:"
    b"foreign-evidence-legacy-fallback"
).hexdigest()

_TRIGGER_SCHEMA = "finite_action_all_parent_trigger"
_OLD_VALUE_SCHEMA = "finite_action_parent_value_scope"
_NEW_ACTION_SCHEMA = "finite_action_declared_intervention"
_TRIGGER_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-all-parent-trigger-schema:v1"
).hexdigest()
_OLD_VALUE_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-parent-value-scope-schema:v1"
).hexdigest()
_NEW_ACTION_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-declared-intervention-schema:v1"
).hexdigest()

_EMPIRICAL_TRIGGER_SCHEMA = "finite_action_empirical_parent_trigger"
_EMPIRICAL_OLD_VALUE_SCHEMA = "finite_action_empirical_parent_values"
_EMPIRICAL_NEW_ACTION_SCHEMA = "finite_action_empirical_transition"
_EMPIRICAL_TRIGGER_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-empirical-parent-trigger-schema:v1"
).hexdigest()
_EMPIRICAL_OLD_VALUE_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-empirical-parent-values-schema:v1"
).hexdigest()
_EMPIRICAL_NEW_ACTION_SCHEMA_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-action-empirical-transition-schema:v1"
).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("portable semantics payload did not freeze to an object")
    return frozen


def _paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(second + ".")
        or first.startswith(second + "[")
        or second.startswith(first + ".")
        or second.startswith(first + "[")
    )


def _exact_path_cover(
    declared: tuple[str, ...],
    observed: tuple[str, ...],
) -> bool:
    return all(
        any(_paths_overlap(path, value) for value in observed) for path in declared
    ) and all(
        any(_paths_overlap(value, path) for path in declared) for value in observed
    )


@dataclass(frozen=True, slots=True)
class PortableFiniteActionInsightSemanticCompiler(CampaignInsightSemanticCompiler):
    """Compile structured card fields into a strong portable local claim."""

    policy_id: str = PORTABLE_FINITE_ACTION_COMPILER_ID
    policy_version: int = PORTABLE_FINITE_ACTION_COMPILER_VERSION
    definition_sha256: str = PORTABLE_FINITE_ACTION_COMPILER_DEFINITION_SHA256

    def compile(
        self,
        *,
        draft: InsightDraft,
        insight_contract: ReflectionInsightContract,
        evidence_lineage: InsightEvidenceLineage,
    ) -> CompiledCampaignInsightSemantics:
        if type(draft) is not InsightDraft:
            raise TypeError("draft must be exact")
        if type(insight_contract) is not ReflectionInsightContract:
            raise TypeError("insight_contract must be exact")
        if type(evidence_lineage) is not InsightEvidenceLineage:
            raise TypeError("evidence_lineage must be exact")
        InsightDraft.__post_init__(draft)
        ReflectionInsightContract.__post_init__(insight_contract)
        InsightEvidenceLineage.__post_init__(evidence_lineage)
        transitions = empirical_finite_action_transitions_for_insight(
            draft,
            evidence_lineage,
        )
        if transitions:
            transition_records = [value.to_record() for value in transitions]
            trigger = TypedEvidencePredicate(
                schema_id=_EMPIRICAL_TRIGGER_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_EMPIRICAL_TRIGGER_SCHEMA_SHA256,
                payload=_object(
                    {
                        "scope": "authenticated_parent_local_value_alternatives",
                        "transitions": transition_records,
                    }
                ),
            )
            old_value_predicate = TypedEvidencePredicate(
                schema_id=_EMPIRICAL_OLD_VALUE_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_EMPIRICAL_OLD_VALUE_SCHEMA_SHA256,
                payload=_object(
                    {
                        "affected_paths": list(draft.affected_paths),
                        "value_constraint": (
                            "one_of_authenticated_parent_local_values"
                        ),
                        "transitions": transition_records,
                    }
                ),
            )
            new_action = TypedEvidencePredicate(
                schema_id=_EMPIRICAL_NEW_ACTION_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_EMPIRICAL_NEW_ACTION_SCHEMA_SHA256,
                payload=_object(
                    {
                        "affected_paths": list(draft.affected_paths),
                        "recommended_option_families": list(
                            draft.recommended_option_families
                        ),
                        "recommended_option_ids": list(
                            draft.recommended_option_ids
                        ),
                        "factor_capabilities": list(draft.factor_capabilities),
                        "matching_law": (
                            "stable_action_and_compiler_with_exact_local_child_value"
                        ),
                        "transitions": transition_records,
                    }
                ),
            )
        else:
            trigger = TypedEvidencePredicate(
                schema_id=_TRIGGER_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_TRIGGER_SCHEMA_SHA256,
                payload=_object(
                    {
                        "scope": "every_parent_in_registered_audit_scope",
                        "prose_trigger_used_for_matching": False,
                    }
                ),
            )
            old_value_predicate = TypedEvidencePredicate(
                schema_id=_OLD_VALUE_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_OLD_VALUE_SCHEMA_SHA256,
                payload=_object(
                    {
                        "affected_paths": list(draft.affected_paths),
                        "value_constraint": "any_pre_intervention_value",
                    }
                ),
            )
            new_action = TypedEvidencePredicate(
                schema_id=_NEW_ACTION_SCHEMA,
                schema_version=1,
                schema_definition_sha256=_NEW_ACTION_SCHEMA_SHA256,
                payload=_object(
                    {
                        "affected_paths": list(draft.affected_paths),
                        "recommended_option_families": list(
                            draft.recommended_option_families
                        ),
                        "recommended_option_ids": list(
                            draft.recommended_option_ids
                        ),
                        "factor_capabilities": list(draft.factor_capabilities),
                        "matching_law": (
                            "exact_bidirectional_path_cover_and_declared_family_id"
                        ),
                    }
                ),
            )
        return CompiledCampaignInsightSemantics(
            draft_content_sha256=draft.content_sha256,
            insight_contract_identity_sha256=insight_contract.identity_sha256,
            evidence_lineage_sha256=evidence_lineage.identity_sha256,
            trigger=trigger,
            old_value_predicate=old_value_predicate,
            new_action=new_action,
            matcher_definition_sha256=(
                PORTABLE_FINITE_ACTION_MATCHER_DEFINITION_SHA256
            ),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )


@dataclass(frozen=True, slots=True)
class PortableFiniteActionHypothesisMatcher:
    """Match compiled finite-action claims to authenticated action records."""

    policy_id: str = PORTABLE_FINITE_ACTION_MATCHER_ID
    policy_version: int = PORTABLE_FINITE_ACTION_MATCHER_VERSION
    definition_sha256: str = PORTABLE_FINITE_ACTION_MATCHER_DEFINITION_SHA256

    @staticmethod
    def _predicate_payload(
        request: GlobalHypothesisAuditRequest,
    ) -> tuple[dict[str, object], tuple[EmpiricalFiniteActionTransition, ...]]:
        trigger = request.trigger
        old_value = request.intervention.old_value_predicate
        new_action = request.intervention.new_action
        expected = (
            (
                trigger.schema_id,
                trigger.schema_version,
                trigger.schema_definition_sha256,
            ),
            (
                old_value.schema_id,
                old_value.schema_version,
                old_value.schema_definition_sha256,
            ),
            (
                new_action.schema_id,
                new_action.schema_version,
                new_action.schema_definition_sha256,
            ),
        )
        required = (
            (_TRIGGER_SCHEMA, 1, _TRIGGER_SCHEMA_SHA256),
            (_OLD_VALUE_SCHEMA, 1, _OLD_VALUE_SCHEMA_SHA256),
            (_NEW_ACTION_SCHEMA, 1, _NEW_ACTION_SCHEMA_SHA256),
        )
        if expected == required:
            payload = thaw_json(new_action.payload)
            if type(payload) is not dict:  # pragma: no cover - closed root.
                raise AssertionError("new-action predicate thawed to a non-object")
            return payload, ()
        empirical_required = (
            (
                _EMPIRICAL_TRIGGER_SCHEMA,
                1,
                _EMPIRICAL_TRIGGER_SCHEMA_SHA256,
            ),
            (
                _EMPIRICAL_OLD_VALUE_SCHEMA,
                1,
                _EMPIRICAL_OLD_VALUE_SCHEMA_SHA256,
            ),
            (
                _EMPIRICAL_NEW_ACTION_SCHEMA,
                1,
                _EMPIRICAL_NEW_ACTION_SCHEMA_SHA256,
            ),
        )
        if expected != empirical_required:
            raise ValueError("audit request uses a foreign finite-action schema")
        payloads = tuple(
            thaw_json(value.payload) for value in (trigger, old_value, new_action)
        )
        if any(type(value) is not dict for value in payloads):
            raise ValueError("empirical finite-action predicates must be objects")
        trigger_payload, old_payload, action_payload = payloads
        transition_records = action_payload.get("transitions")
        if (
            type(transition_records) is not list
            or not transition_records
            or trigger_payload.get("transitions") != transition_records
            or old_payload.get("transitions") != transition_records
        ):
            raise ValueError("empirical finite-action predicates lost their join")
        transitions = tuple(
            EmpiricalFiniteActionTransition.from_record(value)
            for value in transition_records
        )
        if transitions != tuple(
            sorted(transitions, key=lambda value: value.transition_sha256)
        ):
            raise ValueError("empirical finite-action transitions are not canonical")
        return action_payload, transitions

    def classify(
        self,
        request: GlobalHypothesisAuditRequest,
        observation: AuthenticatedHypothesisObservation,
    ) -> HypothesisEvidenceMatchReceipt:
        if type(request) is not GlobalHypothesisAuditRequest:
            raise TypeError("request must be exact")
        if type(observation) is not AuthenticatedHypothesisObservation:
            raise TypeError("observation must be exact")
        GlobalHypothesisAuditRequest.__post_init__(request)
        AuthenticatedHypothesisObservation.__post_init__(observation)
        payload, transitions = self._predicate_payload(request)
        action = thaw_json(observation.observed_action)
        if type(action) is not dict:
            raise ValueError("observed action must be an object")
        required_action_fields = {
            "option_id",
            "option_family",
            "operator_family",
            "changed_paths",
        }
        if not required_action_fields.issubset(action):
            raise ValueError("observed action omits finite-action identity fields")
        declared_paths = tuple(payload.get("affected_paths", ()))
        declared_families = tuple(payload.get("recommended_option_families", ()))
        declared_ids = tuple(payload.get("recommended_option_ids", ()))
        observed_paths = tuple(action["changed_paths"])
        path_exact = _exact_path_cover(declared_paths, observed_paths)
        family_exact = (
            not declared_families or action["option_family"] in declared_families
        )
        option_exact = not declared_ids or action["option_id"] in declared_ids
        operator_exact = action["operator_family"] in (
            request.intervention.admissible_operator_families
        )
        trigger_match = TriggerMatch.EXACT
        if transitions:
            try:
                trigger_match = (
                    TriggerMatch.EXACT
                    if any(
                        transition.parent_matches(observation.parent_configuration)
                        for transition in transitions
                    )
                    else TriggerMatch.OFF_TRIGGER
                )
            except ValueError:
                trigger_match = TriggerMatch.AMBIGUOUS
            action_transition_exact = False
            if path_exact and operator_exact:
                for transition in transitions:
                    if (
                        action["option_id"] == transition.option_id
                        and action["option_family"] == transition.option_family
                        and observation.action_semantics_compiler_id
                        == transition.action_semantics_compiler_id
                        and observation.action_semantics_compiler_version
                        == transition.action_semantics_compiler_version
                        and observation.action_semantics_definition_sha256
                        == transition.action_semantics_definition_sha256
                        and transition.affected_path in observed_paths
                    ):
                        try:
                            if transition.child_matches(
                                observation.child_configuration
                            ):
                                action_transition_exact = True
                                break
                        except ValueError:
                            continue
            family_exact = family_exact and any(
                action["option_family"] == value.option_family
                for value in transitions
            )
            option_exact = option_exact and any(
                action["option_id"] == value.option_id for value in transitions
            )
        else:
            action_transition_exact = path_exact and family_exact and option_exact
        if observation.intervention_identifiability is not (
            InterventionIdentifiability.EXACT_SINGLE
        ):
            intervention_match = InterventionMatch.NON_IDENTIFIABLE
        elif action_transition_exact and operator_exact:
            intervention_match = InterventionMatch.EXACT
        elif any(
            _paths_overlap(declared, observed)
            for declared in declared_paths
            for observed in observed_paths
        ):
            intervention_match = InterventionMatch.NEAR
        else:
            intervention_match = InterventionMatch.DIFFERENT
        return HypothesisEvidenceMatchReceipt(
            request_sha256=request.request_sha256,
            observation_sha256=observation.observation_sha256,
            trigger_match=trigger_match,
            intervention_match=intervention_match,
            matcher_policy_id=self.policy_id,
            matcher_policy_version=self.policy_version,
            matcher_definition_sha256=self.definition_sha256,
        )


__all__ = [
    "PORTABLE_FINITE_ACTION_COMPILER_DEFINITION_SHA256",
    "PORTABLE_FINITE_ACTION_COMPILER_ID",
    "PORTABLE_FINITE_ACTION_COMPILER_VERSION",
    "PORTABLE_FINITE_ACTION_MATCHER_DEFINITION_SHA256",
    "PORTABLE_FINITE_ACTION_MATCHER_ID",
    "PORTABLE_FINITE_ACTION_MATCHER_VERSION",
    "PortableFiniteActionHypothesisMatcher",
    "PortableFiniteActionInsightSemanticCompiler",
]
