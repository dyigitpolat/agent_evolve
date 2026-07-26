"""Authenticated, parent-local semantics for reflected finite actions.

The reflection provider may describe a rule in prose, but executable memory is
derived only from the engine-issued empirical snapshots attached to that rule.
This module projects the trusted direct-single-mutation schema into a small,
workload-neutral transition value: stable action identity plus exact local
parent and child values.  Foreign or legacy evidence remains opaque so callers
can retain their explicitly declared compatibility behavior.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from agent_evolve.application.identifiable_reflection_request import (
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
)
from agent_evolve.application.insight_memory import (
    EmpiricalEvidenceSnapshot,
    InsightEvidenceLineage,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_frozen_json_value,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import InsightDraft


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_TRANSITION_DOMAIN = b"agent-evolve:empirical-finite-action-transition:v1\x00"


def _exact_object(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    return value


def _require_exact_keys(
    value: dict[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} differs from the trusted empirical schema")


def _require_token(value: object, *, name: str, option_id: bool = False) -> str:
    pattern = _OPTION_ID if option_id else _TOKEN
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")
    return value


def _require_path(value: object, *, name: str) -> str:
    if type(value) is not str or _PATH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical rooted JSON path")
    return value


def _require_sha(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    require_sha256(value, name)
    return value


def _path_parts(path: str) -> tuple[str | int, ...]:
    """Parse the closed reflection-path grammar without interpreting prose."""

    _require_path(path, name="path")
    parts: list[str | int] = []
    index = 2
    while index < len(path):
        start = index
        while index < len(path) and path[index] not in ".[":
            index += 1
        if start == index:  # pragma: no cover - the regex closes this branch.
            raise ValueError("path has an empty object key")
        parts.append(path[start:index])
        while index < len(path) and path[index] == "[":
            end = path.index("]", index)
            parts.append(int(path[index + 1 : end]))
            index = end + 1
        if index < len(path):
            if path[index] != ".":  # pragma: no cover - regex closes this.
                raise ValueError("path is malformed")
            index += 1
    return tuple(parts)


def frozen_value_at_canonical_path(
    root: FrozenJsonObject,
    path: str,
) -> FrozenJsonValue:
    """Resolve a trusted canonical path with exact object/array semantics."""

    if type(root) is not FrozenJsonObject or freeze_json(root) is not root:
        raise TypeError("root must be an exact frozen JSON object")
    current: FrozenJsonValue = root
    for part in _path_parts(path):
        if type(part) is str:
            if type(current) is not FrozenJsonObject:
                raise ValueError("object-key path reaches a non-object")
            matches = tuple(value for key, value in current.items if key == part)
            if len(matches) != 1:
                raise ValueError("object-key path does not exist")
            current = matches[0]
        else:
            if type(current) is not FrozenJsonArray:
                raise ValueError("array-index path reaches a non-array")
            if part >= len(current.items):
                raise ValueError("array-index path is out of bounds")
            current = current.items[part]
    return current


@dataclass(frozen=True, slots=True)
class EmpiricalFiniteActionTransition:
    """One authenticated local action occurrence usable as executable memory."""

    contrast_id: str
    source_observation_sha256: str
    source_evidence_id: str
    event_index: int
    workload_instance_sha256: str
    evaluator_contract_sha256: str
    campaign_sha256: str
    option_id: str
    option_identity_sha256: str
    option_family: str
    finite_contract_identity_sha256: str
    affected_path: str
    parent_value: FrozenJsonValue
    child_value: FrozenJsonValue
    parent_configuration_sha256: str
    child_configuration_sha256: str
    action_semantics_compiler_id: str
    action_semantics_compiler_version: int
    action_semantics_definition_sha256: str
    transition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "contrast_id",
            "source_observation_sha256",
            "source_evidence_id",
            "workload_instance_sha256",
            "evaluator_contract_sha256",
            "campaign_sha256",
            "option_identity_sha256",
            "finite_contract_identity_sha256",
            "parent_configuration_sha256",
            "child_configuration_sha256",
            "action_semantics_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _require_token(self.option_id, name="option_id", option_id=True)
        _require_token(self.option_family, name="option_family")
        _require_path(self.affected_path, name="affected_path")
        _require_token(
            self.action_semantics_compiler_id,
            name="action_semantics_compiler_id",
        )
        if (
            type(self.action_semantics_compiler_version) is not int
            or self.action_semantics_compiler_version <= 0
        ):
            raise ValueError("action semantics compiler version must be positive")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be a non-negative exact integer")
        for name in ("parent_value", "child_value"):
            value = getattr(self, name)
            if not is_frozen_json_value(value) or freeze_json(value) is not value:
                raise TypeError(f"{name} must be an exact frozen JSON value")
        if typed_json_equal(self.parent_value, self.child_value):
            raise ValueError("an empirical finite action must change its local value")
        object.__setattr__(
            self,
            "transition_sha256",
            hashlib.sha256(
                _TRANSITION_DOMAIN
                + canonical_typed_json_bytes(freeze_json(self._unsigned_record()))
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contrast_id": self.contrast_id,
            "source_scope": {
                "source_observation_sha256": self.source_observation_sha256,
                "source_evidence_id": self.source_evidence_id,
                "event_index": self.event_index,
                "workload_instance_sha256": self.workload_instance_sha256,
                "evaluator_contract_sha256": self.evaluator_contract_sha256,
                "campaign_sha256": self.campaign_sha256,
            },
            "finite_action": {
                "option_id": self.option_id,
                "option_identity_sha256": self.option_identity_sha256,
                "option_family": self.option_family,
                "finite_contract_identity_sha256": (
                    self.finite_contract_identity_sha256
                ),
            },
            "local_intervention": {
                "affected_path": self.affected_path,
                "parent_value": thaw_json(self.parent_value),
                "child_value": thaw_json(self.child_value),
            },
            "configuration_lineage": {
                "parent_configuration_sha256": self.parent_configuration_sha256,
                "child_configuration_sha256": self.child_configuration_sha256,
            },
            "action_semantics_compiler": {
                "compiler_id": self.action_semantics_compiler_id,
                "compiler_version": self.action_semantics_compiler_version,
                "definition_sha256": self.action_semantics_definition_sha256,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "transition_sha256": self.transition_sha256}

    @classmethod
    def from_record(cls, value: object) -> "EmpiricalFiniteActionTransition":
        record = _exact_object(value, name="transition")
        _require_exact_keys(
            record,
            {
                "schema_version",
                "contrast_id",
                "source_scope",
                "finite_action",
                "local_intervention",
                "configuration_lineage",
                "action_semantics_compiler",
                "transition_sha256",
            },
            name="transition",
        )
        if record["schema_version"] != 1:
            raise ValueError("unsupported empirical transition schema")
        source = _exact_object(record["source_scope"], name="source_scope")
        action = _exact_object(record["finite_action"], name="finite_action")
        local = _exact_object(record["local_intervention"], name="local_intervention")
        lineage = _exact_object(
            record["configuration_lineage"], name="configuration_lineage"
        )
        compiler = _exact_object(
            record["action_semantics_compiler"], name="action_semantics_compiler"
        )
        transition = cls(
            contrast_id=_require_sha(record["contrast_id"], name="contrast_id"),
            source_observation_sha256=_require_sha(
                source.get("source_observation_sha256"),
                name="source_observation_sha256",
            ),
            source_evidence_id=_require_sha(
                source.get("source_evidence_id"), name="source_evidence_id"
            ),
            event_index=source.get("event_index"),  # type: ignore[arg-type]
            workload_instance_sha256=_require_sha(
                source.get("workload_instance_sha256"),
                name="workload_instance_sha256",
            ),
            evaluator_contract_sha256=_require_sha(
                source.get("evaluator_contract_sha256"),
                name="evaluator_contract_sha256",
            ),
            campaign_sha256=_require_sha(
                source.get("campaign_sha256"), name="campaign_sha256"
            ),
            option_id=_require_token(
                action.get("option_id"), name="option_id", option_id=True
            ),
            option_identity_sha256=_require_sha(
                action.get("option_identity_sha256"),
                name="option_identity_sha256",
            ),
            option_family=_require_token(
                action.get("option_family"), name="option_family"
            ),
            finite_contract_identity_sha256=_require_sha(
                action.get("finite_contract_identity_sha256"),
                name="finite_contract_identity_sha256",
            ),
            affected_path=_require_path(
                local.get("affected_path"), name="affected_path"
            ),
            parent_value=freeze_json(local.get("parent_value")),
            child_value=freeze_json(local.get("child_value")),
            parent_configuration_sha256=_require_sha(
                lineage.get("parent_configuration_sha256"),
                name="parent_configuration_sha256",
            ),
            child_configuration_sha256=_require_sha(
                lineage.get("child_configuration_sha256"),
                name="child_configuration_sha256",
            ),
            action_semantics_compiler_id=_require_token(
                compiler.get("compiler_id"), name="action_semantics_compiler_id"
            ),
            action_semantics_compiler_version=compiler.get(  # type: ignore[arg-type]
                "compiler_version"
            ),
            action_semantics_definition_sha256=_require_sha(
                compiler.get("definition_sha256"),
                name="action_semantics_definition_sha256",
            ),
        )
        if transition.to_record() != record:
            raise ValueError("empirical transition record is not canonical")
        return transition

    def parent_matches(self, configuration: FrozenJsonObject) -> bool:
        """Return whether the authenticated local precondition still holds.

        This is deliberately weaker than source-context identity.  It is
        suitable for deciding whether an action can be described as the same
        local intervention, but it must not by itself authorize transfer of an
        observed optimization benefit to a different configuration.
        """

        return typed_json_equal(
            frozen_value_at_canonical_path(configuration, self.affected_path),
            self.parent_value,
        )

    def exact_parent_matches(self, configuration: FrozenJsonObject) -> bool:
        """Return whether the complete current parent is the observed parent."""

        if type(configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        return typed_json_sha256(configuration) == self.parent_configuration_sha256

    def child_matches(self, configuration: FrozenJsonObject) -> bool:
        return typed_json_equal(
            frozen_value_at_canonical_path(configuration, self.affected_path),
            self.child_value,
        )


def _transition_from_snapshot(
    snapshot: EmpiricalEvidenceSnapshot,
) -> EmpiricalFiniteActionTransition | None:
    snapshot.__post_init__()
    trusted_identity = (
        snapshot.fact_schema_id,
        snapshot.fact_schema_version,
        snapshot.fact_schema_definition_sha256,
    )
    expected_identity = (
        IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
        IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
        IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    )
    if trusted_identity != expected_identity:
        return None
    facts = _exact_object(thaw_json(snapshot.facts), name="empirical facts")
    _require_exact_keys(
        facts,
        {
            "schema_version",
            "design_kind",
            "comparison_anchor",
            "mechanism_identifying_design",
            "permitted_insight_kinds",
            "request_binding",
            "source_scope",
            "occurrence_lineage",
            "finite_action",
            "local_intervention",
            "configuration_lineage",
            "outcome_lineage",
            "observed_metric_effects",
        },
        name="empirical facts",
    )
    if (
        facts["schema_version"] != 1
        or facts["design_kind"] != "direct_single_mutation"
        or facts["comparison_anchor"] != "current_parent"
        or facts["mechanism_identifying_design"] is not False
        or facts["permitted_insight_kinds"] != ["empirical_predictive_rule"]
    ):
        raise ValueError("trusted empirical facts contradict their schema")
    request_binding = _exact_object(
        facts["request_binding"], name="request_binding"
    )
    compiler = _exact_object(
        request_binding.get("action_semantics_compiler"),
        name="action_semantics_compiler",
    )
    _require_exact_keys(
        compiler,
        {"compiler_id", "compiler_version", "definition_sha256"},
        name="action_semantics_compiler",
    )
    compiler_definition = _require_sha(
        compiler["definition_sha256"], name="action_semantics_definition_sha256"
    )
    if snapshot.action_semantics_definition_sha256 != compiler_definition:
        raise ValueError("snapshot and fact action semantics identities differ")
    source = _exact_object(facts["source_scope"], name="source_scope")
    action = _exact_object(facts["finite_action"], name="finite_action")
    local = _exact_object(facts["local_intervention"], name="local_intervention")
    lineage = _exact_object(
        facts["configuration_lineage"], name="configuration_lineage"
    )
    transition = EmpiricalFiniteActionTransition(
        contrast_id=snapshot.contrast_id,
        source_observation_sha256=_require_sha(
            source.get("source_observation_sha256"),
            name="source_observation_sha256",
        ),
        source_evidence_id=_require_sha(
            source.get("source_evidence_id"), name="source_evidence_id"
        ),
        event_index=source.get("event_index"),  # type: ignore[arg-type]
        workload_instance_sha256=_require_sha(
            source.get("workload_instance_sha256"),
            name="workload_instance_sha256",
        ),
        evaluator_contract_sha256=_require_sha(
            source.get("evaluator_contract_sha256"),
            name="evaluator_contract_sha256",
        ),
        campaign_sha256=_require_sha(
            source.get("campaign_sha256"), name="campaign_sha256"
        ),
        option_id=_require_token(
            action.get("option_id"), name="option_id", option_id=True
        ),
        option_identity_sha256=_require_sha(
            action.get("option_identity_sha256"), name="option_identity_sha256"
        ),
        option_family=_require_token(
            action.get("option_family"), name="option_family"
        ),
        finite_contract_identity_sha256=_require_sha(
            action.get("finite_contract_identity_sha256"),
            name="finite_contract_identity_sha256",
        ),
        affected_path=_require_path(
            local.get("affected_path"), name="affected_path"
        ),
        parent_value=freeze_json(local.get("parent_value")),
        child_value=freeze_json(local.get("child_value")),
        parent_configuration_sha256=_require_sha(
            lineage.get("parent_configuration_sha256"),
            name="parent_configuration_sha256",
        ),
        child_configuration_sha256=_require_sha(
            lineage.get("child_configuration_sha256"),
            name="child_configuration_sha256",
        ),
        action_semantics_compiler_id=_require_token(
            compiler.get("compiler_id"), name="action_semantics_compiler_id"
        ),
        action_semantics_compiler_version=compiler.get(  # type: ignore[arg-type]
            "compiler_version"
        ),
        action_semantics_definition_sha256=compiler_definition,
    )
    return transition


def empirical_finite_action_transitions(
    evidence_lineage: InsightEvidenceLineage,
) -> tuple[EmpiricalFiniteActionTransition, ...]:
    """Project only recognized direct evidence; foreign schemas stay legacy."""

    if type(evidence_lineage) is not InsightEvidenceLineage:
        raise TypeError("evidence_lineage must be exact")
    evidence_lineage.__post_init__()
    projected = tuple(
        transition
        for snapshot in evidence_lineage.empirical_evidence
        if (transition := _transition_from_snapshot(snapshot)) is not None
    )
    canonical = tuple(sorted(projected, key=lambda value: value.transition_sha256))
    if len({value.transition_sha256 for value in canonical}) != len(canonical):
        raise ValueError("empirical lineage repeats a finite action transition")
    return canonical


def empirical_finite_action_transitions_for_insight(
    draft: InsightDraft,
    evidence_lineage: InsightEvidenceLineage,
) -> tuple[EmpiricalFiniteActionTransition, ...]:
    """Join trusted transitions to the model's closed structured declaration."""

    if type(draft) is not InsightDraft:
        raise TypeError("draft must be exact")
    draft.__post_init__()
    transitions = empirical_finite_action_transitions(evidence_lineage)
    for transition in transitions:
        if transition.contrast_id not in draft.evidence_contrast_ids:
            raise ValueError("empirical transition is not cited by the insight")
        if transition.affected_path not in draft.affected_paths:
            raise ValueError("empirical transition path differs from the insight")
        if transition.option_family not in draft.recommended_option_families:
            raise ValueError("empirical transition family differs from the insight")
        if (
            draft.recommended_option_ids
            and transition.option_id not in draft.recommended_option_ids
        ):
            raise ValueError("empirical transition action differs from the insight")
    return transitions


__all__ = [
    "EmpiricalFiniteActionTransition",
    "empirical_finite_action_transitions",
    "empirical_finite_action_transitions_for_insight",
    "frozen_value_at_canonical_path",
]
