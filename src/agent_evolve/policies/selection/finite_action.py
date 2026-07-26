"""Prospective task-keyed uniform comparator for matched finite action sets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.finite_action_selection import (
    EngineFiniteActionRequest,
    FiniteActionDecision,
    FiniteActionSelectorKind,
    ProspectiveUniformRankToken,
    validate_finite_action_decision,
)


POLICY_ID = "task_keyed_uniform_finite_action"
POLICY_VERSION = 2
_DEFINITION_DOMAIN = b"agent-evolve:task-keyed-uniform-finite-action:def:v2\x00"
_STATE_DOMAIN = b"agent-evolve:task-keyed-uniform-finite-action:state:v2\x00"
_RANK_DOMAIN = b"agent-evolve:task-keyed-uniform-finite-action:rank:v2\x00"
_DEFINITION = {
    "rank": (
        "counter-domain-separated SHA-256 rejection sampling followed by "
        "modulo K"
    ),
    "propensity": "1/K under a prospectively randomized uniform seed",
    "resample_on_model_alias": False,
    "current_outcome_access": False,
}
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    _DEFINITION_DOMAIN
    + json.dumps(
        _DEFINITION,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskKeyedUniformFiniteActionPolicy:
    """Freeze and replay one uniform ordinal without inspecting outcomes."""

    schedule_seed_sha256: str
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(
        init=False,
        default=POLICY_DEFINITION_SHA256,
    )
    schedule_state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.schedule_seed_sha256, "schedule_seed_sha256")
        object.__setattr__(
            self,
            "schedule_state_sha256",
            hashlib.sha256(
                _STATE_DOMAIN + bytes.fromhex(self.schedule_seed_sha256)
            ).hexdigest(),
        )

    def _selected_ordinal(
        self,
        authority: FiniteActionSetAuthority,
        *,
        task_sha256: str,
    ) -> int:
        FiniteActionSetAuthority.__post_init__(authority)
        require_sha256(task_sha256, "task_sha256")
        framing = (
            _RANK_DOMAIN
            + bytes.fromhex(self.schedule_seed_sha256)
            + bytes.fromhex(task_sha256)
            + bytes.fromhex(authority.authority_sha256)
            + bytes.fromhex(authority.support.support_sha256)
            + bytes.fromhex(authority.card.card_content_sha256)
        )
        cardinality = authority.support.cardinality
        sample_space = 1 << 256
        acceptance_limit = sample_space - (sample_space % cardinality)
        # Rejection sampling is required because ``2**256 % K`` is non-zero
        # for K=5, 6, and 7.  A direct digest modulo K would therefore make the
        # recorded 1/K propensity false outside the first K=8 release.
        for counter in range(1 << 32):
            digest = hashlib.sha256(
                framing + counter.to_bytes(4, "big", signed=False)
            ).digest()
            sample = int.from_bytes(digest, "big", signed=False)
            if sample < acceptance_limit:
                return sample % cardinality
        raise RuntimeError("uniform finite-action rejection sampler exhausted")

    def freeze_rank(
        self,
        authority: FiniteActionSetAuthority,
        *,
        task_sha256: str,
        pre_outcome_phase_commit_sha256: str,
    ) -> ProspectiveUniformRankToken:
        if type(authority) is not FiniteActionSetAuthority:
            raise TypeError("authority must be an exact FiniteActionSetAuthority")
        FiniteActionSetAuthority.__post_init__(authority)
        require_sha256(
            pre_outcome_phase_commit_sha256,
            "pre_outcome_phase_commit_sha256",
        )
        return ProspectiveUniformRankToken(
            task_sha256=task_sha256,
            authority_sha256=authority.authority_sha256,
            support_sha256=authority.support.support_sha256,
            card_reference=authority.card.reference,
            card_content_sha256=authority.card.card_content_sha256,
            cardinality=authority.support.cardinality,
            selected_ordinal=self._selected_ordinal(
                authority,
                task_sha256=task_sha256,
            ),
            schedule_policy_id=self.policy_id,
            schedule_policy_version=self.policy_version,
            schedule_definition_sha256=self.definition_sha256,
            schedule_state_sha256=self.schedule_state_sha256,
            pre_outcome_phase_commit_sha256=pre_outcome_phase_commit_sha256,
            current_outcome_access=False,
        )

    def choose(self, request: EngineFiniteActionRequest) -> FiniteActionDecision:
        if type(request) is not EngineFiniteActionRequest:
            raise TypeError("request must be an exact EngineFiniteActionRequest")
        EngineFiniteActionRequest.__post_init__(request)
        token = request.prospective_rank
        expected_policy = (
            self.policy_id,
            self.policy_version,
            self.definition_sha256,
            self.schedule_state_sha256,
        )
        observed_policy = (
            token.schedule_policy_id,
            token.schedule_policy_version,
            token.schedule_definition_sha256,
            token.schedule_state_sha256,
        )
        if observed_policy != expected_policy:
            raise ValueError("prospective rank token uses a foreign schedule policy")
        if token.selected_ordinal != self._selected_ordinal(
            request.authority,
            task_sha256=token.task_sha256,
        ):
            raise ValueError("prospective rank token does not replay its public rank")
        row = request.authority.support.options[token.selected_ordinal]
        decision = FiniteActionDecision(
            authority_sha256=request.authority.authority_sha256,
            support_sha256=request.authority.support.support_sha256,
            selector_kind=FiniteActionSelectorKind.ENGINE,
            selected_ordinal=token.selected_ordinal,
            option_id=row.option.option_id,
            option_identity_sha256=row.option.identity_sha256,
            child_configuration_sha256=row.option.child_configuration_sha256,
            phenotype_identity_sha256=row.phenotype_identity_sha256,
            selector_policy_id=self.policy_id,
            selector_policy_version=self.policy_version,
            selector_definition_sha256=self.definition_sha256,
            prospective_token_sha256=token.token_sha256,
            model_call_id=None,
            model_prompt_sha256=None,
            model_telemetry_sha256=None,
            propensity_numerator=1,
            propensity_denominator=request.authority.support.cardinality,
            current_outcome_access=False,
        )
        validate_finite_action_decision(request.authority, decision)
        return decision


__all__ = [
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "TaskKeyedUniformFiniteActionPolicy",
]
