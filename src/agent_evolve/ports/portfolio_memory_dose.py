"""Typed, workload-neutral contracts for bounded portfolio memory use.

Showing a memory card to a selector and claiming that a particular action was
supported by that card are different events.  A single provider call exposes
the whole response slate to every prompt-visible card, so an uncited member is
*not* a blinded control.  It is nevertheless useful to reserve uncited
exploration slots and to require every explicit card attribution to name an
action that trusted application code declared compatible before the call.

This module records those narrower claims.  It owns no prompt renderer,
provider, benchmark, evaluator, memory bank, or allocation policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_CONTRACT_DOMAIN = b"agent-evolve:bounded-portfolio-memory-dose-contract:v1\x00"
_MEMBER_DOMAIN = b"agent-evolve:portfolio-memory-dose-member:v1\x00"
_ASSESSMENT_DOMAIN = b"agent-evolve:portfolio-memory-dose-assessment:v1\x00"

BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID = "bounded_relevance_aware_memory_dose"
BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION = 1
BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:bounded-relevance-aware-memory-dose:v1;"
    b"prompt-wide-exposure-not-blinding;exact-card-coverage;"
    b"bounded-supported-member-count;bounded-cards-per-member;"
    b"precommitted-card-option-compatibility;proposal-evaluation-join"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_option_id(value: str, *, name: str) -> None:
    if type(value) is not str or _OPTION_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed option grammar")


def _canonical_card_keys(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError(f"{name} must be an exact tuple of card keys")
    for value in values:
        _require_token(value, name=name)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


class PortfolioMemoryDoseStage(str, Enum):
    """The two outcome-blind boundaries at which dose is checked."""

    PROPOSED_SLATE = "proposed_slate"
    EVALUATED_PORTFOLIO = "evaluated_portfolio"


class PortfolioMemoryExposureScope(str, Enum):
    """What the dose receipt can honestly say about prompt exposure."""

    PROMPT_WIDE = "prompt_wide"


class PortfolioMemoryDoseViolation(str, Enum):
    """Closed failure reasons suitable for schema repair and durable traces."""

    FOREIGN_CARD_ATTRIBUTION = "foreign_card_attribution"
    INCOMPATIBLE_CARD_ACTION = "incompatible_card_action"
    TOO_MANY_CARDS_PER_MEMBER = "too_many_cards_per_member"
    ASSIGNED_CARD_OMITTED = "assigned_card_omitted"
    TOO_FEW_SUPPORTED_MEMBERS = "too_few_supported_members"
    TOO_MANY_SUPPORTED_MEMBERS = "too_many_supported_members"
    TOO_FEW_UNATTRIBUTED_MEMBERS = "too_few_unattributed_members"
    EVALUATED_MEMBER_NOT_IN_PROPOSAL = "evaluated_member_not_in_proposal"


@dataclass(frozen=True, slots=True)
class PortfolioMemoryDoseCardSupport:
    """Pre-call exact finite options to which one card may be attributed."""

    card_key: str
    card_content_sha256: str
    finite_contract_identity_sha256: str
    compatible_options: tuple[tuple[str, str], ...]
    support_policy_id: str
    support_policy_version: int
    support_policy_definition_sha256: str

    def __post_init__(self) -> None:
        _require_token(self.card_key, name="card_key")
        for name in (
            "card_content_sha256",
            "finite_contract_identity_sha256",
            "support_policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.compatible_options) is not tuple or not self.compatible_options:
            raise ValueError("compatible_options must be a non-empty exact tuple")
        option_ids: list[str] = []
        option_identities: list[str] = []
        for item in self.compatible_options:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("compatible_options must contain exact pairs")
            option_id, option_identity_sha256 = item
            _require_option_id(option_id, name="compatible option_id")
            require_sha256(option_identity_sha256, "compatible option identity")
            option_ids.append(option_id)
            option_identities.append(option_identity_sha256)
        if self.compatible_options != tuple(sorted(set(self.compatible_options))):
            raise ValueError("compatible_options must be unique and canonical")
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("compatible_options cannot repeat an option_id")
        if len(set(option_identities)) != len(option_identities):
            raise ValueError("compatible_options cannot repeat an option identity")
        _require_token(self.support_policy_id, name="support_policy_id")
        if type(self.support_policy_version) is not int or self.support_policy_version <= 0:
            raise ValueError("support_policy_version must be positive")

    def supports(self, option_id: str, option_identity_sha256: str) -> bool:
        self.__post_init__()
        return (option_id, option_identity_sha256) in self.compatible_options

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "card_key": self.card_key,
            "card_content_sha256": self.card_content_sha256,
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "compatible_options": [
                {
                    "option_id": option_id,
                    "option_identity_sha256": option_identity_sha256,
                }
                for option_id, option_identity_sha256 in self.compatible_options
            ],
            "support_policy": {
                "policy_id": self.support_policy_id,
                "policy_version": self.support_policy_version,
                "definition_sha256": self.support_policy_definition_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class BoundedPortfolioMemoryDoseContract:
    """Precommitted bounds for explicit card-supported proposal/evaluation slots.

    ``supported`` always means an explicit, compatibility-checked attribution.
    It never means that other members were hidden from the card.  True card
    blinding requires a separate prompt-redacted provider call.
    """

    card_supports: tuple[PortfolioMemoryDoseCardSupport, ...]
    proposed_supported_member_bounds: tuple[int, int]
    evaluated_supported_member_bounds: tuple[int, int]
    minimum_unattributed_proposed_members: int
    minimum_unattributed_evaluated_members: int
    maximum_cards_per_member: int = 1
    require_every_assigned_card: bool = True
    exposure_scope: PortfolioMemoryExposureScope = (
        PortfolioMemoryExposureScope.PROMPT_WIDE
    )
    policy_id: str = BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID
    policy_version: int = BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION
    policy_definition_sha256: str = (
        BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256
    )
    contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.card_supports) is not tuple
            or not self.card_supports
            or any(
                type(value) is not PortfolioMemoryDoseCardSupport
                for value in self.card_supports
            )
        ):
            raise ValueError("card_supports must contain exact support values")
        for value in self.card_supports:
            PortfolioMemoryDoseCardSupport.__post_init__(value)
        keys = tuple(value.card_key for value in self.card_supports)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("card_supports must use unique canonical card order")
        contract_ids = {
            value.finite_contract_identity_sha256 for value in self.card_supports
        }
        if len(contract_ids) != 1:
            raise ValueError("all card supports must bind one finite contract")
        for name in (
            "proposed_supported_member_bounds",
            "evaluated_supported_member_bounds",
        ):
            bounds = getattr(self, name)
            if (
                type(bounds) is not tuple
                or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
            ):
                raise TypeError(f"{name} must be an exact integer pair")
            lower, upper = bounds
            if lower < 0 or upper < lower:
                raise ValueError(f"{name} must be ordered and non-negative")
            if self.require_every_assigned_card and lower < len(keys):
                raise ValueError(
                    f"{name} lower bound cannot cover every assigned card"
                )
        for name in (
            "minimum_unattributed_proposed_members",
            "minimum_unattributed_evaluated_members",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if type(self.maximum_cards_per_member) is not int or (
            self.maximum_cards_per_member <= 0
        ):
            raise ValueError("maximum_cards_per_member must be positive")
        if type(self.require_every_assigned_card) is not bool:
            raise TypeError("require_every_assigned_card must be exact bool")
        if self.exposure_scope is not PortfolioMemoryExposureScope.PROMPT_WIDE:
            raise ValueError("bounded dose v1 records prompt-wide exposure only")
        if (
            self.policy_id != BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID
            or self.policy_version != BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION
            or self.policy_definition_sha256
            != BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("unsupported bounded memory-dose policy")
        object.__setattr__(
            self,
            "contract_sha256",
            _hash(_CONTRACT_DOMAIN, self._unsigned_record()),
        )

    @property
    def assigned_card_keys(self) -> tuple[str, ...]:
        return tuple(value.card_key for value in self.card_supports)

    @property
    def finite_contract_identity_sha256(self) -> str:
        return self.card_supports[0].finite_contract_identity_sha256

    def support_for(self, card_key: str) -> PortfolioMemoryDoseCardSupport:
        self.__post_init__()
        for value in self.card_supports:
            if value.card_key == card_key:
                return value
        raise ValueError("card_key is outside the bounded dose contract")

    def bounds_for(self, stage: PortfolioMemoryDoseStage) -> tuple[int, int]:
        if stage is PortfolioMemoryDoseStage.PROPOSED_SLATE:
            return self.proposed_supported_member_bounds
        if stage is PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO:
            return self.evaluated_supported_member_bounds
        raise TypeError("stage must be exact PortfolioMemoryDoseStage")

    def minimum_unattributed_for(self, stage: PortfolioMemoryDoseStage) -> int:
        if stage is PortfolioMemoryDoseStage.PROPOSED_SLATE:
            return self.minimum_unattributed_proposed_members
        if stage is PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO:
            return self.minimum_unattributed_evaluated_members
        raise TypeError("stage must be exact PortfolioMemoryDoseStage")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "card_supports": [value.to_record() for value in self.card_supports],
            "proposed_supported_member_bounds": list(
                self.proposed_supported_member_bounds
            ),
            "evaluated_supported_member_bounds": list(
                self.evaluated_supported_member_bounds
            ),
            "minimum_unattributed_proposed_members": (
                self.minimum_unattributed_proposed_members
            ),
            "minimum_unattributed_evaluated_members": (
                self.minimum_unattributed_evaluated_members
            ),
            "maximum_cards_per_member": self.maximum_cards_per_member,
            "require_every_assigned_card": self.require_every_assigned_card,
            "exposure_scope": self.exposure_scope.value,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "unattributed_member_interpretation": (
                "prompt_exposed_exploration_not_blinded_control"
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "contract_sha256": self.contract_sha256}


@dataclass(frozen=True, slots=True)
class PortfolioMemoryDoseMember:
    """One proposed or evaluated finite option with explicit card attribution."""

    rank: int
    option_id: str
    option_identity_sha256: str
    supporting_card_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be positive")
        _require_option_id(self.option_id, name="option_id")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        _canonical_card_keys(self.supporting_card_keys, name="supporting_card_keys")

    def _content_record(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "supporting_card_keys": list(self.supporting_card_keys),
        }

    @property
    def content_binding_sha256(self) -> str:
        self.__post_init__()
        return _hash(_MEMBER_DOMAIN, self._content_record())

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "rank": self.rank,
            **self._content_record(),
            "content_binding_sha256": self.content_binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class PortfolioMemoryDoseAssessment:
    """Content-free, replayable result of one proposal/evaluation dose check."""

    contract_sha256: str
    stage: PortfolioMemoryDoseStage
    member_content_binding_sha256s: tuple[str, ...]
    supported_member_ranks: tuple[int, ...]
    unattributed_member_ranks: tuple[int, ...]
    card_attribution_ranks: tuple[tuple[str, tuple[int, ...]], ...]
    violations: tuple[PortfolioMemoryDoseViolation, ...]
    proposal_assessment_sha256: str | None = None
    assessment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "contract_sha256")
        if type(self.stage) is not PortfolioMemoryDoseStage:
            raise TypeError("stage must be exact PortfolioMemoryDoseStage")
        if (
            type(self.member_content_binding_sha256s) is not tuple
            or not self.member_content_binding_sha256s
        ):
            raise ValueError("member bindings must be a non-empty exact tuple")
        for value in self.member_content_binding_sha256s:
            require_sha256(value, "member content binding")
        ranks = tuple(range(1, len(self.member_content_binding_sha256s) + 1))
        for name in ("supported_member_ranks", "unattributed_member_ranks"):
            values = getattr(self, name)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
            if any(type(value) is not int or value not in ranks for value in values):
                raise ValueError(f"{name} contains an invalid rank")
        if set(self.supported_member_ranks).intersection(
            self.unattributed_member_ranks
        ) or set(self.supported_member_ranks).union(
            self.unattributed_member_ranks
        ) != set(ranks):
            raise ValueError("supported/unattributed ranks must partition members")
        if type(self.card_attribution_ranks) is not tuple:
            raise TypeError("card_attribution_ranks must be an exact tuple")
        keys: list[str] = []
        for item in self.card_attribution_ranks:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("card_attribution_ranks must contain exact pairs")
            card_key, card_ranks = item
            _require_token(card_key, name="card attribution key")
            if (
                type(card_ranks) is not tuple
                or card_ranks != tuple(sorted(set(card_ranks)))
                or any(value not in ranks for value in card_ranks)
            ):
                raise ValueError("card attribution ranks must be canonical")
            keys.append(card_key)
        if keys != sorted(set(keys)):
            raise ValueError("card_attribution_ranks must use canonical card order")
        if type(self.violations) is not tuple or any(
            type(value) is not PortfolioMemoryDoseViolation for value in self.violations
        ):
            raise TypeError("violations must contain exact closed values")
        if self.violations != tuple(sorted(set(self.violations), key=lambda x: x.value)):
            raise ValueError("violations must be unique and canonical")
        if self.stage is PortfolioMemoryDoseStage.PROPOSED_SLATE:
            if self.proposal_assessment_sha256 is not None:
                raise ValueError("proposal assessment cannot name an earlier proposal")
        elif self.proposal_assessment_sha256 is None:
            raise ValueError("evaluated assessment must join its proposal")
        else:
            require_sha256(
                self.proposal_assessment_sha256,
                "proposal_assessment_sha256",
            )
        object.__setattr__(
            self,
            "assessment_sha256",
            _hash(_ASSESSMENT_DOMAIN, self._unsigned_record()),
        )

    @property
    def passed(self) -> bool:
        return not self.violations

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "contract_sha256": self.contract_sha256,
            "stage": self.stage.value,
            "member_content_binding_sha256s": list(
                self.member_content_binding_sha256s
            ),
            "supported_member_ranks": list(self.supported_member_ranks),
            "unattributed_member_ranks": list(self.unattributed_member_ranks),
            "card_attribution_ranks": [
                {"card_key": key, "member_ranks": list(ranks)}
                for key, ranks in self.card_attribution_ranks
            ],
            "violations": [value.value for value in self.violations],
            "proposal_assessment_sha256": self.proposal_assessment_sha256,
            "exposure_scope": PortfolioMemoryExposureScope.PROMPT_WIDE.value,
            "unattributed_members_are_blinded_controls": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "passed": self.passed,
            "assessment_sha256": self.assessment_sha256,
        }


class PortfolioMemoryDoseRejected(ValueError):
    """A proposed or evaluated slate violated its sealed dose contract."""

    def __init__(self, assessment: PortfolioMemoryDoseAssessment) -> None:
        if type(assessment) is not PortfolioMemoryDoseAssessment or assessment.passed:
            raise ValueError("rejection requires a failed exact assessment")
        super().__init__("portfolio violates its bounded memory-dose contract")
        self.assessment = assessment


def _assess(
    contract: BoundedPortfolioMemoryDoseContract,
    members: tuple[PortfolioMemoryDoseMember, ...],
    *,
    stage: PortfolioMemoryDoseStage,
    proposal_assessment: PortfolioMemoryDoseAssessment | None,
) -> PortfolioMemoryDoseAssessment:
    if type(contract) is not BoundedPortfolioMemoryDoseContract:
        raise TypeError("contract must be exact BoundedPortfolioMemoryDoseContract")
    contract.__post_init__()
    if (
        type(members) is not tuple
        or not members
        or any(type(value) is not PortfolioMemoryDoseMember for value in members)
    ):
        raise ValueError("members must contain exact dose members")
    for value in members:
        PortfolioMemoryDoseMember.__post_init__(value)
    if tuple(value.rank for value in members) != tuple(range(1, len(members) + 1)):
        raise ValueError("members must use contiguous rank order")
    if len({value.option_id for value in members}) != len(members):
        raise ValueError("members cannot repeat an option_id")
    if len({value.option_identity_sha256 for value in members}) != len(members):
        raise ValueError("members cannot repeat an option identity")
    if type(stage) is not PortfolioMemoryDoseStage:
        raise TypeError("stage must be exact PortfolioMemoryDoseStage")
    if stage is PortfolioMemoryDoseStage.PROPOSED_SLATE:
        if proposal_assessment is not None:
            raise ValueError("proposal stage cannot join an earlier proposal")
    else:
        if type(proposal_assessment) is not PortfolioMemoryDoseAssessment:
            raise TypeError("evaluated stage requires an exact proposal assessment")
        PortfolioMemoryDoseAssessment.__post_init__(proposal_assessment)
        if (
            proposal_assessment.stage is not PortfolioMemoryDoseStage.PROPOSED_SLATE
            or proposal_assessment.contract_sha256 != contract.contract_sha256
            or not proposal_assessment.passed
        ):
            raise ValueError("evaluated stage requires a passing matching proposal")

    violations: set[PortfolioMemoryDoseViolation] = set()
    assigned = set(contract.assigned_card_keys)
    support_by_key = {value.card_key: value for value in contract.card_supports}
    card_ranks = {key: [] for key in contract.assigned_card_keys}
    supported_ranks: list[int] = []
    unattributed_ranks: list[int] = []
    proposal_bindings = (
        set()
        if proposal_assessment is None
        else set(proposal_assessment.member_content_binding_sha256s)
    )
    for member in members:
        keys = member.supporting_card_keys
        if keys:
            supported_ranks.append(member.rank)
        else:
            unattributed_ranks.append(member.rank)
        if len(keys) > contract.maximum_cards_per_member:
            violations.add(PortfolioMemoryDoseViolation.TOO_MANY_CARDS_PER_MEMBER)
        for card_key in keys:
            if card_key not in assigned:
                violations.add(PortfolioMemoryDoseViolation.FOREIGN_CARD_ATTRIBUTION)
                continue
            card_ranks[card_key].append(member.rank)
            if not support_by_key[card_key].supports(
                member.option_id,
                member.option_identity_sha256,
            ):
                violations.add(PortfolioMemoryDoseViolation.INCOMPATIBLE_CARD_ACTION)
        if proposal_assessment is not None and (
            member.content_binding_sha256 not in proposal_bindings
        ):
            violations.add(
                PortfolioMemoryDoseViolation.EVALUATED_MEMBER_NOT_IN_PROPOSAL
            )

    if contract.require_every_assigned_card and any(
        not card_ranks[key] for key in contract.assigned_card_keys
    ):
        violations.add(PortfolioMemoryDoseViolation.ASSIGNED_CARD_OMITTED)
    lower, upper = contract.bounds_for(stage)
    if len(supported_ranks) < lower:
        violations.add(PortfolioMemoryDoseViolation.TOO_FEW_SUPPORTED_MEMBERS)
    if len(supported_ranks) > upper:
        violations.add(PortfolioMemoryDoseViolation.TOO_MANY_SUPPORTED_MEMBERS)
    if len(unattributed_ranks) < contract.minimum_unattributed_for(stage):
        violations.add(PortfolioMemoryDoseViolation.TOO_FEW_UNATTRIBUTED_MEMBERS)
    return PortfolioMemoryDoseAssessment(
        contract_sha256=contract.contract_sha256,
        stage=stage,
        member_content_binding_sha256s=tuple(
            value.content_binding_sha256 for value in members
        ),
        supported_member_ranks=tuple(supported_ranks),
        unattributed_member_ranks=tuple(unattributed_ranks),
        card_attribution_ranks=tuple(
            (key, tuple(card_ranks[key])) for key in contract.assigned_card_keys
        ),
        violations=tuple(sorted(violations, key=lambda value: value.value)),
        proposal_assessment_sha256=(
            None
            if proposal_assessment is None
            else proposal_assessment.assessment_sha256
        ),
    )


def assess_proposed_portfolio_memory_dose(
    contract: BoundedPortfolioMemoryDoseContract,
    members: tuple[PortfolioMemoryDoseMember, ...],
) -> PortfolioMemoryDoseAssessment:
    """Assess the complete provider-authored slate before allocation."""

    return _assess(
        contract,
        members,
        stage=PortfolioMemoryDoseStage.PROPOSED_SLATE,
        proposal_assessment=None,
    )


def assess_evaluated_portfolio_memory_dose(
    contract: BoundedPortfolioMemoryDoseContract,
    members: tuple[PortfolioMemoryDoseMember, ...],
    *,
    proposal_assessment: PortfolioMemoryDoseAssessment,
) -> PortfolioMemoryDoseAssessment:
    """Assess the allocated portfolio and join it to the sealed proposal."""

    return _assess(
        contract,
        members,
        stage=PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO,
        proposal_assessment=proposal_assessment,
    )


def require_passing_portfolio_memory_dose(
    assessment: PortfolioMemoryDoseAssessment,
) -> None:
    """Raise a typed rejection while retaining the complete failure receipt."""

    if type(assessment) is not PortfolioMemoryDoseAssessment:
        raise TypeError("assessment must be exact PortfolioMemoryDoseAssessment")
    PortfolioMemoryDoseAssessment.__post_init__(assessment)
    if not assessment.passed:
        raise PortfolioMemoryDoseRejected(assessment)


__all__ = [
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256",
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID",
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION",
    "BoundedPortfolioMemoryDoseContract",
    "PortfolioMemoryDoseAssessment",
    "PortfolioMemoryDoseCardSupport",
    "PortfolioMemoryDoseMember",
    "PortfolioMemoryDoseRejected",
    "PortfolioMemoryDoseStage",
    "PortfolioMemoryDoseViolation",
    "PortfolioMemoryExposureScope",
    "assess_evaluated_portfolio_memory_dose",
    "assess_proposed_portfolio_memory_dose",
    "require_passing_portfolio_memory_dose",
]
