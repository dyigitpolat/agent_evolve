"""Trusted derivation of card-to-finite-option support for bounded memory dose."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.application.finite_action_transition import (
    EmpiricalFiniteActionTransition,
    empirical_finite_action_transitions_for_insight,
)
from agent_evolve.application.insight_memory import InsightEvidenceLineage
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import typed_json_sha256
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import InsightDraft
from agent_evolve.ports.portfolio_memory_dose import PortfolioMemoryDoseCardSupport


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")

PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID = "semantic_finite_option_memory_support"
PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION = 2
PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantic-finite-option-memory-support:v2;"
    b"trusted-card-content;current-finite-contract;family-match;"
    b"all-parent-relative-changed-paths-within-card-paths;"
    b"optional-exact-recommended-option-filter;"
    b"authenticated-direct-evidence-requires-stable-action-and-exact-local-"
    b"parent-child-transition;foreign-evidence-legacy-fallback"
).hexdigest()
EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID = (
    "exact_parent_semantic_finite_option_memory_support"
)
EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION = 1
EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:exact-parent-semantic-finite-option-memory-support:v1;"
    b"trusted-card-content;current-finite-contract;family-and-path-match;"
    b"authenticated-direct-transition;complete-source-parent-sha256-match;"
    b"exact-local-child-transition;forced-transfer-authority-only"
).hexdigest()
PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID = "portfolio_memory_context_transfer"
PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION = 1
PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-context-transfer:v1;"
    b"authenticated-transition-source-parents;current-parent-sha256;"
    b"exact-source-parent-authorizes-action-replay;"
    b"local-intervention-match-is-advisory-only;provider-and-outcome-blind"
).hexdigest()
_CONTEXT_TRANSFER_DOMAIN = b"agent-evolve:portfolio-memory-context-transfer:v1\x00"
PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID = (
    "portfolio_memory_typed_transfer_ladder"
)
PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION = 1
PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:portfolio-memory-typed-transfer-ladder:v1;"
    b"exact-source-parent-and-action=executable-replay;"
    b"same-local-precondition-and-action=advisory-only;"
    b"affected-path-and-action-family=uncalibrated-advisory-only;"
    b"unsupported=no-delivery;causal-memory-credit=false;"
    b"provider-and-outcome-blind=true"
).hexdigest()
_TRANSFER_LADDER_DOMAIN = (
    b"agent-evolve:portfolio-memory-typed-transfer-ladder:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


class PortfolioMemoryDoseSupportScope(str, Enum):
    """How much source context is required before an action is executable."""

    LOCAL_INTERVENTION = "local_intervention"
    EXACT_SOURCE_PARENT = "exact_source_parent"


class PortfolioMemoryTransferTier(str, Enum):
    """Closed authority levels for transferring one empirical action rule."""

    EXACT_ACTION_REPLAY = "exact_action_replay"
    LOCAL_ACTION_ADVISORY = "local_action_advisory"
    PATH_FAMILY_ADVISORY = "path_family_advisory"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class PortfolioMemoryTransferLadderAssessment:
    """Typed transfer evidence without silently converting analogy into action.

    ``exact_replay_option_ids`` are the only options that may be forced by a
    bounded dose.  The two advisory sets are prompt evidence only.  In
    particular, a path/family match deliberately relaxes the historical option
    identity and is labelled uncalibrated until a future held-out transfer model
    earns a quantitative shrinkage law.

    Retrieval is not a causal memory experiment.  Consequently this assessment
    never authorizes card-level causal credit, including at the exact tier.
    """

    card_key: str
    card_content_sha256: str
    finite_contract_identity_sha256: str
    current_parent_configuration_sha256: str
    source_transition_sha256s: tuple[str, ...]
    exact_replay_option_ids: tuple[str, ...]
    local_advisory_option_ids: tuple[str, ...]
    path_family_advisory_option_ids: tuple[str, ...]
    tier: PortfolioMemoryTransferTier
    assessment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed token grammar")
        for name in (
            "card_content_sha256",
            "finite_contract_identity_sha256",
            "current_parent_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "source_transition_sha256s",
            "exact_replay_option_ids",
            "local_advisory_option_ids",
            "path_family_advisory_option_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
            for value in values:
                if name == "source_transition_sha256s":
                    require_sha256(value, "source_transition_sha256")
                elif type(value) is not str or _OPTION_ID.fullmatch(value) is None:
                    raise ValueError(f"{name} contains an invalid option ID")
        if type(self.tier) is not PortfolioMemoryTransferTier:
            raise TypeError("tier must be exact PortfolioMemoryTransferTier")
        expected_tier = (
            PortfolioMemoryTransferTier.EXACT_ACTION_REPLAY
            if self.exact_replay_option_ids
            else PortfolioMemoryTransferTier.LOCAL_ACTION_ADVISORY
            if self.local_advisory_option_ids
            else PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY
            if self.path_family_advisory_option_ids
            else PortfolioMemoryTransferTier.UNSUPPORTED
        )
        if self.tier is not expected_tier:
            raise ValueError("transfer tier differs from its authenticated support")
        if set(self.exact_replay_option_ids).difference(
            self.local_advisory_option_ids
        ):
            raise ValueError("exact replay options must also have local support")
        object.__setattr__(
            self,
            "assessment_sha256",
            hashlib.sha256(
                _TRANSFER_LADDER_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def exact_action_replay_authorized(self) -> bool:
        return self.tier is PortfolioMemoryTransferTier.EXACT_ACTION_REPLAY

    @property
    def advisory_delivery_authorized(self) -> bool:
        return self.tier in {
            PortfolioMemoryTransferTier.LOCAL_ACTION_ADVISORY,
            PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY,
        }

    @property
    def causal_memory_credit_authorized(self) -> bool:
        return False

    @property
    def deliverable_option_ids(self) -> tuple[str, ...]:
        if self.exact_replay_option_ids:
            return self.exact_replay_option_ids
        if self.local_advisory_option_ids:
            return self.local_advisory_option_ids
        return self.path_family_advisory_option_ids

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "card_key": self.card_key,
            "card_content_sha256": self.card_content_sha256,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "current_parent_configuration_sha256": (
                self.current_parent_configuration_sha256
            ),
            "source_transition_sha256s": list(self.source_transition_sha256s),
            "exact_replay_option_ids": list(self.exact_replay_option_ids),
            "local_advisory_option_ids": list(self.local_advisory_option_ids),
            "path_family_advisory_option_ids": list(
                self.path_family_advisory_option_ids
            ),
            "tier": self.tier.value,
            "exact_action_replay_authorized": (
                self.exact_action_replay_authorized
            ),
            "advisory_delivery_authorized": self.advisory_delivery_authorized,
            "causal_memory_credit_authorized": False,
            "transfer_calibration": (
                "not_applicable_exact_replay"
                if self.exact_action_replay_authorized
                else "uncalibrated_advisory"
                if self.advisory_delivery_authorized
                else "not_deliverable"
            ),
            "historical_option_identity_relaxed": (
                self.tier is PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY
            ),
            "policy": {
                "policy_id": PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID,
                "policy_version": PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION,
                "definition_sha256": (
                    PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256
                ),
            },
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "assessment_sha256": self.assessment_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryContextTransferAssessment:
    """Outcome-blind authority check for transferring one empirical action.

    An authenticated local transition remains useful as advisory evidence when
    its local precondition matches.  Hard replay authority is narrower: at
    present it is granted only when the complete current parent equals an
    observed source parent.  Later learned transfer models can implement a new
    versioned policy instead of silently weakening this boundary.
    """

    card_key: str
    card_content_sha256: str
    finite_contract_identity_sha256: str
    current_parent_configuration_sha256: str
    source_parent_configuration_sha256s: tuple[str, ...]
    local_intervention_support_available: bool
    exact_source_parent_match: bool
    exact_action_replay_authorized: bool
    assessment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed token grammar")
        for name in (
            "card_content_sha256",
            "finite_contract_identity_sha256",
            "current_parent_configuration_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.source_parent_configuration_sha256s) is not tuple
            or self.source_parent_configuration_sha256s
            != tuple(sorted(set(self.source_parent_configuration_sha256s)))
        ):
            raise ValueError("source parent hashes must be unique and canonical")
        for value in self.source_parent_configuration_sha256s:
            require_sha256(value, "source_parent_configuration_sha256")
        for name in (
            "local_intervention_support_available",
            "exact_source_parent_match",
            "exact_action_replay_authorized",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.exact_source_parent_match != (
            self.current_parent_configuration_sha256
            in self.source_parent_configuration_sha256s
        ):
            raise ValueError("exact source-parent verdict differs from its hashes")
        if self.exact_action_replay_authorized != (
            self.local_intervention_support_available
            and self.exact_source_parent_match
        ):
            raise ValueError("exact replay authority differs from trusted evidence")
        object.__setattr__(
            self,
            "assessment_sha256",
            hashlib.sha256(
                _CONTEXT_TRANSFER_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "card_key": self.card_key,
            "card_content_sha256": self.card_content_sha256,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "current_parent_configuration_sha256": (
                self.current_parent_configuration_sha256
            ),
            "source_parent_configuration_sha256s": list(
                self.source_parent_configuration_sha256s
            ),
            "local_intervention_support_available": (
                self.local_intervention_support_available
            ),
            "exact_source_parent_match": self.exact_source_parent_match,
            "exact_action_replay_authorized": self.exact_action_replay_authorized,
            "transfer_authority": (
                "exact_source_parent_action_replay"
                if self.exact_action_replay_authorized
                else "advisory_only"
            ),
            "policy": {
                "policy_id": PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID,
                "policy_version": PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION,
                "definition_sha256": (
                    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256
                ),
            },
            "provider_fields_consulted": False,
            "outcome_values_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "assessment_sha256": self.assessment_sha256}


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported JSON path segment")
    return "".join(parts)


def _path_segments(path: str) -> tuple[str, ...]:
    if type(path) is not str or not path.startswith("$."):
        raise ValueError("affected paths must be canonical object-root JSON paths")
    # Existing semantic contracts use object paths in this workload-neutral
    # boundary.  Treat array notation as a segment delimiter without parsing
    # values; descendant containment remains conservative and deterministic.
    body = path[2:].replace("[", ".[")
    parts = tuple(value for value in body.split(".") if value)
    if not parts:
        raise ValueError("affected paths cannot name the root")
    return parts


def _changed_path_is_within_affected_path(changed: str, affected: str) -> bool:
    changed_segments = _path_segments(changed)
    affected_segments = _path_segments(affected)
    return changed_segments[: len(affected_segments)] == affected_segments


@dataclass(frozen=True, slots=True)
class PortfolioMemoryDoseCardSemantics:
    """Trusted card semantics needed to derive current-palette compatibility."""

    card_key: str
    card_content_sha256: str
    affected_paths: tuple[str, ...]
    recommended_option_families: tuple[str, ...]
    recommended_option_ids: tuple[str, ...] = ()
    empirical_transitions: tuple[EmpiricalFiniteActionTransition, ...] = ()
    support_scope: PortfolioMemoryDoseSupportScope = (
        PortfolioMemoryDoseSupportScope.LOCAL_INTERVENTION
    )

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed token grammar")
        require_sha256(self.card_content_sha256, "card_content_sha256")
        if (
            type(self.affected_paths) is not tuple
            or not self.affected_paths
            or self.affected_paths != tuple(sorted(set(self.affected_paths)))
        ):
            raise ValueError("affected_paths must be non-empty and canonical")
        for value in self.affected_paths:
            _path_segments(value)
        if (
            type(self.recommended_option_families) is not tuple
            or not self.recommended_option_families
            or self.recommended_option_families
            != tuple(sorted(set(self.recommended_option_families)))
        ):
            raise ValueError(
                "recommended_option_families must be non-empty and canonical"
            )
        for value in self.recommended_option_families:
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise ValueError("recommended option family is invalid")
        if (
            type(self.recommended_option_ids) is not tuple
            or self.recommended_option_ids
            != tuple(sorted(set(self.recommended_option_ids)))
        ):
            raise ValueError("recommended_option_ids must be canonical")
        for value in self.recommended_option_ids:
            if type(value) is not str or _OPTION_ID.fullmatch(value) is None:
                raise ValueError("recommended option ID is invalid")
        if type(self.empirical_transitions) is not tuple or any(
            type(value) is not EmpiricalFiniteActionTransition
            for value in self.empirical_transitions
        ):
            raise TypeError(
                "empirical_transitions must contain exact transition values"
            )
        if type(self.support_scope) is not PortfolioMemoryDoseSupportScope:
            raise TypeError("support_scope must be exact PortfolioMemoryDoseSupportScope")
        for value in self.empirical_transitions:
            value.__post_init__()
            if value.affected_path not in self.affected_paths:
                raise ValueError("empirical transition path differs from the card")
            if value.option_family not in self.recommended_option_families:
                raise ValueError("empirical transition family differs from the card")
            if (
                self.recommended_option_ids
                and value.option_id not in self.recommended_option_ids
            ):
                raise ValueError("empirical transition action differs from the card")
        if self.empirical_transitions != tuple(
            sorted(
                self.empirical_transitions,
                key=lambda value: value.transition_sha256,
            )
        ):
            raise ValueError("empirical_transitions must be canonical")
        if len(
            {value.transition_sha256 for value in self.empirical_transitions}
        ) != len(self.empirical_transitions):
            raise ValueError("empirical_transitions must be unique")

    @classmethod
    def from_insight(
        cls,
        *,
        card_key: str,
        card_content_sha256: str,
        draft: InsightDraft,
        evidence_lineage: InsightEvidenceLineage,
        support_scope: PortfolioMemoryDoseSupportScope = (
            PortfolioMemoryDoseSupportScope.LOCAL_INTERVENTION
        ),
    ) -> "PortfolioMemoryDoseCardSemantics":
        """Build executable card semantics from structured, trusted lineage."""

        if type(draft) is not InsightDraft:
            raise TypeError("draft must be exact")
        if type(evidence_lineage) is not InsightEvidenceLineage:
            raise TypeError("evidence_lineage must be exact")
        draft.__post_init__()
        evidence_lineage.__post_init__()
        return cls(
            card_key=card_key,
            card_content_sha256=card_content_sha256,
            affected_paths=tuple(sorted(set(draft.affected_paths))),
            recommended_option_families=tuple(
                sorted(set(draft.recommended_option_families))
            ),
            recommended_option_ids=tuple(
                sorted(set(draft.recommended_option_ids))
            ),
            empirical_transitions=(
                empirical_finite_action_transitions_for_insight(
                    draft,
                    evidence_lineage,
                )
            ),
            support_scope=support_scope,
        )


def assess_portfolio_memory_context_transfer(
    semantics: PortfolioMemoryDoseCardSemantics,
    contract: FiniteVariationContract,
) -> PortfolioMemoryContextTransferAssessment:
    """Assess source-context authority without consuming provider outputs."""

    if type(semantics) is not PortfolioMemoryDoseCardSemantics:
        raise TypeError("semantics must be exact PortfolioMemoryDoseCardSemantics")
    semantics.__post_init__()
    validate_finite_variation_contract(contract)
    source_parents = tuple(
        sorted(
            {
                value.parent_configuration_sha256
                for value in semantics.empirical_transitions
            }
        )
    )
    current_parent = typed_json_sha256(contract.parent_configuration)
    local_support = False
    if semantics.empirical_transitions:
        for option in contract.options:
            if any(
                transition.option_id == option.option_id
                and transition.option_family == option.family
                and transition.parent_matches(contract.parent_configuration)
                and transition.child_matches(option.child_configuration)
                for transition in semantics.empirical_transitions
            ):
                local_support = True
                break
    exact_match = current_parent in source_parents
    return PortfolioMemoryContextTransferAssessment(
        card_key=semantics.card_key,
        card_content_sha256=semantics.card_content_sha256,
        finite_contract_identity_sha256=contract.identity_sha256,
        current_parent_configuration_sha256=current_parent,
        source_parent_configuration_sha256s=source_parents,
        local_intervention_support_available=local_support,
        exact_source_parent_match=exact_match,
        exact_action_replay_authorized=local_support and exact_match,
    )


def assess_portfolio_memory_transfer_ladder(
    semantics: PortfolioMemoryDoseCardSemantics,
    contract: FiniteVariationContract,
) -> PortfolioMemoryTransferLadderAssessment:
    """Resolve exact, local, and path/family transfer authority in one pass.

    The first two tiers require authenticated empirical transitions.  The
    path/family tier uses only the card's declared affected paths and action
    families against the current finite contract.  It is useful for exposing a
    mechanism after parent drift, but it cannot force an action or receive
    causal credit.
    """

    if type(semantics) is not PortfolioMemoryDoseCardSemantics:
        raise TypeError("semantics must be exact PortfolioMemoryDoseCardSemantics")
    semantics.__post_init__()
    validate_finite_variation_contract(contract)
    base = CandidateId("candidate_memory_transfer_parent")
    target = CandidateId("candidate_memory_transfer_child")
    path_family_ids: list[str] = []
    local_ids: list[str] = []
    exact_ids: list[str] = []
    for option in contract.options:
        if option.family not in semantics.recommended_option_families:
            continue
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=base,
            target_candidate_id=target,
        )
        changed_paths = tuple(_path_text(value.path) for value in patch.operations)
        if not changed_paths or not all(
            any(
                _changed_path_is_within_affected_path(changed, affected)
                for affected in semantics.affected_paths
            )
            for changed in changed_paths
        ):
            continue
        path_family_ids.append(option.option_id)
        matching_transitions = tuple(
            transition
            for transition in semantics.empirical_transitions
            if transition.option_id == option.option_id
            and transition.option_family == option.family
            and changed_paths == (transition.affected_path,)
            and transition.parent_matches(contract.parent_configuration)
            and transition.child_matches(option.child_configuration)
        )
        if not matching_transitions:
            continue
        local_ids.append(option.option_id)
        if any(
            transition.exact_parent_matches(contract.parent_configuration)
            for transition in matching_transitions
        ):
            exact_ids.append(option.option_id)
    exact = tuple(sorted(set(exact_ids)))
    local = tuple(sorted(set(local_ids)))
    path_family = tuple(sorted(set(path_family_ids)))
    tier = (
        PortfolioMemoryTransferTier.EXACT_ACTION_REPLAY
        if exact
        else PortfolioMemoryTransferTier.LOCAL_ACTION_ADVISORY
        if local
        else PortfolioMemoryTransferTier.PATH_FAMILY_ADVISORY
        if path_family
        else PortfolioMemoryTransferTier.UNSUPPORTED
    )
    return PortfolioMemoryTransferLadderAssessment(
        card_key=semantics.card_key,
        card_content_sha256=semantics.card_content_sha256,
        finite_contract_identity_sha256=contract.identity_sha256,
        current_parent_configuration_sha256=typed_json_sha256(
            contract.parent_configuration
        ),
        source_transition_sha256s=tuple(
            sorted(
                transition.transition_sha256
                for transition in semantics.empirical_transitions
            )
        ),
        exact_replay_option_ids=exact,
        local_advisory_option_ids=local,
        path_family_advisory_option_ids=path_family,
        tier=tier,
    )


def derive_portfolio_memory_advisory_card_support(
    semantics: PortfolioMemoryDoseCardSemantics,
    contract: FiniteVariationContract,
) -> PortfolioMemoryDoseCardSupport:
    """Project the ladder's deliverable options into existing support tooling.

    This support object is for deterministic card/lane matching and prompt
    attribution.  Callers must consult the paired ladder assessment before
    deciding whether it can become a hard bounded dose; advisory tiers remain
    non-executable by construction.
    """

    assessment = assess_portfolio_memory_transfer_ladder(semantics, contract)
    if not assessment.deliverable_option_ids:
        raise ValueError("card has no deliverable transfer tier in the finite contract")
    identity_by_id = {
        option.option_id: option.identity_sha256 for option in contract.options
    }
    return PortfolioMemoryDoseCardSupport(
        card_key=semantics.card_key,
        card_content_sha256=semantics.card_content_sha256,
        finite_contract_identity_sha256=contract.identity_sha256,
        compatible_options=tuple(
            (option_id, identity_by_id[option_id])
            for option_id in assessment.deliverable_option_ids
        ),
        support_policy_id=PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID,
        support_policy_version=PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION,
        support_policy_definition_sha256=(
            PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256
        ),
    )


def derive_portfolio_memory_dose_card_support(
    semantics: PortfolioMemoryDoseCardSemantics,
    contract: FiniteVariationContract,
) -> PortfolioMemoryDoseCardSupport:
    """Seal the exact current-parent actions compatible with one card.

    Every changed path must overlap a declared affected path.  This prevents a
    multi-coordinate option from being attributed to a single-coordinate card
    merely because one of its edits happens to overlap.
    """

    if type(semantics) is not PortfolioMemoryDoseCardSemantics:
        raise TypeError("semantics must be exact PortfolioMemoryDoseCardSemantics")
    PortfolioMemoryDoseCardSemantics.__post_init__(semantics)
    validate_finite_variation_contract(contract)
    if (
        semantics.support_scope is PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT
        and not semantics.empirical_transitions
    ):
        raise ValueError(
            "exact source-parent support requires authenticated empirical transitions"
        )
    recommended_ids = set(semantics.recommended_option_ids)
    available_ids = {value.option_id for value in contract.options}
    if not recommended_ids.issubset(available_ids):
        raise ValueError("recommended option IDs escape the finite contract")
    base = CandidateId("candidate_memory_dose_parent")
    target = CandidateId("candidate_memory_dose_child")
    compatible: list[tuple[str, str]] = []
    for option in contract.options:
        if option.family not in semantics.recommended_option_families:
            continue
        if recommended_ids and option.option_id not in recommended_ids:
            continue
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=base,
            target_candidate_id=target,
        )
        changed_paths = tuple(_path_text(value.path) for value in patch.operations)
        if not changed_paths:
            continue
        paths_compatible = all(
            any(
                _changed_path_is_within_affected_path(changed, affected)
                for affected in semantics.affected_paths
            )
            for changed in changed_paths
        )
        if not paths_compatible:
            continue
        if semantics.empirical_transitions and not any(
            transition.option_id == option.option_id
            and transition.option_family == option.family
            and changed_paths == (transition.affected_path,)
            and transition.parent_matches(contract.parent_configuration)
            and (
                semantics.support_scope
                is PortfolioMemoryDoseSupportScope.LOCAL_INTERVENTION
                or transition.exact_parent_matches(contract.parent_configuration)
            )
            and transition.child_matches(option.child_configuration)
            for transition in semantics.empirical_transitions
        ):
            continue
        if paths_compatible:
            compatible.append((option.option_id, option.identity_sha256))
    if not compatible:
        raise ValueError("card has no compatible action in the finite contract")
    if (
        semantics.support_scope is PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT
    ):
        support_policy_id = EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID
        support_policy_version = (
            EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION
        )
        support_policy_definition_sha256 = (
            EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256
        )
    else:
        support_policy_id = PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID
        support_policy_version = PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION
        support_policy_definition_sha256 = (
            PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256
        )
    return PortfolioMemoryDoseCardSupport(
        card_key=semantics.card_key,
        card_content_sha256=semantics.card_content_sha256,
        finite_contract_identity_sha256=contract.identity_sha256,
        compatible_options=tuple(sorted(compatible)),
        support_policy_id=support_policy_id,
        support_policy_version=support_policy_version,
        support_policy_definition_sha256=support_policy_definition_sha256,
    )


__all__ = [
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256",
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID",
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION",
    "PortfolioMemoryContextTransferAssessment",
    "PortfolioMemoryDoseCardSemantics",
    "PortfolioMemoryDoseSupportScope",
    "PortfolioMemoryTransferLadderAssessment",
    "PortfolioMemoryTransferTier",
    "assess_portfolio_memory_context_transfer",
    "assess_portfolio_memory_transfer_ladder",
    "derive_portfolio_memory_advisory_card_support",
    "derive_portfolio_memory_dose_card_support",
]
