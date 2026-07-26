"""Model-independent candidate pools for clean cross-model reranking assays.

The calibrated selector historically exposed one feasibility witness whose
ordering was keyed by the complete provider request.  Model execution settings
therefore changed the candidate slate before a model answered.  This module
owns the replacement scientific control: an oversubscribed pool keyed only by
the optimization task, replicate seed, wave, parent, finite contract, and hard
allocation constraints.

The policy never reads a model profile, call ID, prompt hash, provider setting,
option outcome, or prior evaluator value.  A model may rank and forecast the
pool, but may not change its membership.  Open proposal generation remains a
separate acquisition mode so reranking and generation capability can be
identified independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_witness,
)


POLICY_ID = "task_keyed_common_candidate_pool"
POLICY_VERSION = 6
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:task-keyed-common-candidate-pool:v6;"
    b"sampling-state=replicate-seed,benchmark,wave,parent,finite-contract,constraints,"
    b"required-option-ids;"
    b"sampling-state-excludes=universe-size,model-selection-size;"
    b"decision-task-adds=universe-size,model-selection-size;"
    b"candidate-universe=model-independent;outcomes=false;provider-fields=false;"
    b"model-selection-size=authenticated;universe-size=authenticated;"
    b"nested-universe-prefix-law=true;presentation-relative-order=state-keyed;"
    b"feasible-evaluation-subset=engine-certified;"
    b"certificate-members-not-prefix-positioned;presentation=independently-state-keyed;"
    b"required-options-enter-membership-without-presentation-priority;"
    b"complete-finite-contract-mode-resolves-universe-size-at-selection"
).hexdigest()

_STATE_DOMAIN = b"agent-evolve:common-candidate-pool-state:v1\x00"
_TASK_DOMAIN = b"agent-evolve:common-candidate-pool-task:v2\x00"
_OPTION_ORDER_DOMAIN = b"agent-evolve:common-candidate-pool-option-order:v2\x00"
_PRESENTATION_ORDER_DOMAIN = (
    b"agent-evolve:common-candidate-pool-presentation-order:v2\x00"
)
_DECISION_DOMAIN = b"agent-evolve:common-candidate-pool-decision:v2\x00"
_MAX_SEED = (1 << 63) - 1


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


def _state_record(
    *,
    replicate_seed: int,
    benchmark_sha256: str,
    wave_index: int,
    parent_configuration_sha256: str,
    finite_contract_sha256: str,
    evaluation_size: int,
    min_distinct_families: int | None,
    require_pairwise_disjoint_parent_patches: bool,
    required_option_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "replicate_seed": replicate_seed,
        "benchmark_sha256": benchmark_sha256,
        "wave_index": wave_index,
        "parent_configuration_sha256": parent_configuration_sha256,
        "finite_contract_sha256": finite_contract_sha256,
        "evaluation_size": evaluation_size,
        "min_distinct_families": min_distinct_families,
        "require_pairwise_disjoint_parent_patches": (
            require_pairwise_disjoint_parent_patches
        ),
        "required_option_ids": list(required_option_ids),
    }


def _task_record(
    *,
    replicate_seed: int,
    benchmark_sha256: str,
    wave_index: int,
    parent_configuration_sha256: str,
    finite_contract_sha256: str,
    candidate_pool_size: int,
    model_selection_size: int,
    evaluation_size: int,
    min_distinct_families: int | None,
    require_pairwise_disjoint_parent_patches: bool,
    required_option_ids: tuple[str, ...],
) -> dict[str, object]:
    state = _state_record(
        replicate_seed=replicate_seed,
        benchmark_sha256=benchmark_sha256,
        wave_index=wave_index,
        parent_configuration_sha256=parent_configuration_sha256,
        finite_contract_sha256=finite_contract_sha256,
        evaluation_size=evaluation_size,
        min_distinct_families=min_distinct_families,
        require_pairwise_disjoint_parent_patches=(
            require_pairwise_disjoint_parent_patches
        ),
        required_option_ids=required_option_ids,
    )
    return {
        **state,
        "schema_version": 3,
        "state_identity_sha256": _hash(_STATE_DOMAIN, state),
        "candidate_pool_size": candidate_pool_size,
        "model_selection_size": model_selection_size,
    }


def _validate_inputs(
    *,
    replicate_seed: int,
    benchmark_sha256: str,
    wave_index: int,
    parent_configuration_sha256: str,
    contract: FiniteVariationContract,
    candidate_pool_size: int,
    model_selection_size: int,
    evaluation_size: int,
    min_distinct_families: int | None,
    require_pairwise_disjoint_parent_patches: bool,
    required_option_ids: tuple[str, ...],
) -> None:
    if type(replicate_seed) is not int or not 0 <= replicate_seed <= _MAX_SEED:
        raise ValueError("replicate_seed must be an exact non-negative int63")
    require_sha256(benchmark_sha256, "benchmark_sha256")
    require_sha256(parent_configuration_sha256, "parent_configuration_sha256")
    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    if contract.parent_configuration_sha256 != parent_configuration_sha256:
        raise ValueError("common-pool parent differs from the finite contract")
    if type(wave_index) is not int or wave_index <= 0:
        raise ValueError("wave_index must be a positive exact integer")
    if type(candidate_pool_size) is not int or candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be a positive exact integer")
    if type(model_selection_size) is not int or not (
        0 < model_selection_size <= candidate_pool_size
    ):
        raise ValueError(
            "model_selection_size must lie in [1, candidate_pool_size]"
        )
    if type(evaluation_size) is not int or not (
        0 < evaluation_size <= model_selection_size
    ):
        raise ValueError("evaluation_size must lie in [1, model_selection_size]")
    if candidate_pool_size > len(contract.options):
        raise ValueError("candidate_pool_size exceeds the finite option count")
    if min_distinct_families is not None and (
        type(min_distinct_families) is not int
        or not 1 <= min_distinct_families <= evaluation_size
    ):
        raise ValueError(
            "min_distinct_families must lie within the evaluation size"
        )
    if type(require_pairwise_disjoint_parent_patches) is not bool:
        raise TypeError(
            "require_pairwise_disjoint_parent_patches must be an exact bool"
        )
    if type(required_option_ids) is not tuple or any(
        type(value) is not str for value in required_option_ids
    ):
        raise TypeError("required_option_ids must be an exact string tuple")
    if required_option_ids != tuple(sorted(set(required_option_ids))):
        raise ValueError("required_option_ids must be unique and canonical")
    available = {value.option_id for value in contract.options}
    if not set(required_option_ids).issubset(available):
        raise ValueError("required option IDs escape the finite contract")
    if len(required_option_ids) > candidate_pool_size:
        raise ValueError("required option IDs exceed the candidate-pool size")


@dataclass(frozen=True, slots=True)
class CommonCandidatePoolDecision:
    """Authenticated exact pool membership and its model-free derivation."""

    replicate_seed: int
    benchmark_sha256: str
    wave_index: int
    parent_configuration_sha256: str
    finite_contract_sha256: str
    candidate_pool_size: int
    model_selection_size: int
    evaluation_size: int
    min_distinct_families: int | None
    require_pairwise_disjoint_parent_patches: bool
    required_option_ids: tuple[str, ...]
    option_ids: tuple[str, ...]
    feasibility_witness_option_ids: tuple[str, ...]
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    policy_definition_sha256: str = POLICY_DEFINITION_SHA256
    state_identity_sha256: str = field(init=False)
    task_identity_sha256: str = field(init=False)
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.replicate_seed) is not int or not (
            0 <= self.replicate_seed <= _MAX_SEED
        ):
            raise ValueError("replicate_seed must be an exact non-negative int63")
        require_sha256(self.benchmark_sha256, "benchmark_sha256")
        require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        require_sha256(self.finite_contract_sha256, "finite_contract_sha256")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        if self.policy_id != POLICY_ID or self.policy_version != POLICY_VERSION:
            raise ValueError("common-pool policy identity drifted")
        if self.policy_definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("common-pool policy definition drifted")
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("wave_index must be a positive exact integer")
        if type(self.candidate_pool_size) is not int or self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be a positive exact integer")
        if type(self.model_selection_size) is not int or not (
            0 < self.model_selection_size <= self.candidate_pool_size
        ):
            raise ValueError(
                "model_selection_size must lie in [1, candidate_pool_size]"
            )
        if type(self.evaluation_size) is not int or not (
            0 < self.evaluation_size <= self.model_selection_size
        ):
            raise ValueError("evaluation_size must lie in [1, model_selection_size]")
        if type(self.option_ids) is not tuple or len(self.option_ids) != (
            self.candidate_pool_size
        ):
            raise ValueError("option_ids must contain the exact candidate-pool size")
        if len(set(self.option_ids)) != len(self.option_ids):
            raise ValueError("common candidate pool cannot repeat an option")
        if type(self.required_option_ids) is not tuple or (
            self.required_option_ids
            != tuple(sorted(set(self.required_option_ids)))
        ):
            raise ValueError("required_option_ids must be unique and canonical")
        if not set(self.required_option_ids).issubset(self.option_ids):
            raise ValueError("required options must enter the common pool")
        if type(self.feasibility_witness_option_ids) is not tuple:
            raise TypeError("feasibility_witness_option_ids must be an exact tuple")
        if len(self.feasibility_witness_option_ids) != self.evaluation_size:
            raise ValueError("feasibility witness must cover the evaluation size")
        if not set(self.feasibility_witness_option_ids).issubset(self.option_ids):
            raise ValueError("feasibility witness escapes the common pool")
        state_identity = _hash(_STATE_DOMAIN, self._state_record())
        task_identity = _hash(_TASK_DOMAIN, self._task_record())
        object.__setattr__(self, "state_identity_sha256", state_identity)
        object.__setattr__(self, "task_identity_sha256", task_identity)
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    def _state_record(self) -> dict[str, object]:
        return _state_record(
            replicate_seed=self.replicate_seed,
            benchmark_sha256=self.benchmark_sha256,
            wave_index=self.wave_index,
            parent_configuration_sha256=self.parent_configuration_sha256,
            finite_contract_sha256=self.finite_contract_sha256,
            evaluation_size=self.evaluation_size,
            min_distinct_families=self.min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                self.require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=self.required_option_ids,
        )

    def _task_record(self) -> dict[str, object]:
        return _task_record(
            replicate_seed=self.replicate_seed,
            benchmark_sha256=self.benchmark_sha256,
            wave_index=self.wave_index,
            parent_configuration_sha256=self.parent_configuration_sha256,
            finite_contract_sha256=self.finite_contract_sha256,
            candidate_pool_size=self.candidate_pool_size,
            model_selection_size=self.model_selection_size,
            evaluation_size=self.evaluation_size,
            min_distinct_families=self.min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                self.require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=self.required_option_ids,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "state": self._state_record(),
            "state_identity_sha256": _hash(
                _STATE_DOMAIN,
                self._state_record(),
            ),
            "task": self._task_record(),
            "task_identity_sha256": _hash(_TASK_DOMAIN, self._task_record()),
            "option_ids": list(self.option_ids),
            "required_option_ids": list(self.required_option_ids),
            "feasibility_witness_option_ids": list(
                self.feasibility_witness_option_ids
            ),
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "model_or_provider_fields_consulted": False,
            "objective_or_outcome_values_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def to_prompt_record(self) -> dict[str, object]:
        """Return model-visible membership without the hidden feasible subset."""

        self.__post_init__()
        return {
            "schema_version": 3,
            "state_identity_sha256": self.state_identity_sha256,
            "task_identity_sha256": self.task_identity_sha256,
            "decision_sha256": self.decision_sha256,
            "option_ids": list(self.option_ids),
            "candidate_pool_size": self.candidate_pool_size,
            "model_selection_size": self.model_selection_size,
            "membership_is_fixed": True,
            "model_must_select_and_rank_exact_count": self.model_selection_size,
            "model_or_provider_fields_consulted": False,
            "objective_or_outcome_values_consulted": False,
        }


@dataclass(frozen=True, slots=True)
class TaskKeyedCommonCandidatePoolPolicy:
    """Construct one exact candidate universe independently of the model."""

    replicate_seed: int
    candidate_pool_size: int | None = 8
    model_selection_size: int = 8
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    policy_definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.replicate_seed) is not int or not (
            0 <= self.replicate_seed <= _MAX_SEED
        ):
            raise ValueError("replicate_seed must be an exact non-negative int63")
        if self.candidate_pool_size is not None and (
            type(self.candidate_pool_size) is not int
            or self.candidate_pool_size <= 0
        ):
            raise ValueError(
                "candidate_pool_size must be a positive exact integer or None"
            )
        if type(self.model_selection_size) is not int or (
            self.model_selection_size <= 0
            or (
                self.candidate_pool_size is not None
                and self.model_selection_size > self.candidate_pool_size
            )
        ):
            raise ValueError(
                "model_selection_size must be positive and fit the candidate pool"
            )
        if self.policy_id != POLICY_ID or self.policy_version != POLICY_VERSION:
            raise ValueError("common-pool policy identity drifted")
        if self.policy_definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("common-pool policy definition drifted")

    def select(
        self,
        *,
        benchmark_sha256: str,
        wave_index: int,
        parent_configuration_sha256: str,
        contract: FiniteVariationContract,
        evaluation_size: int,
        min_distinct_families: int | None,
        require_pairwise_disjoint_parent_patches: bool,
        required_option_ids: tuple[str, ...] = (),
    ) -> CommonCandidatePoolDecision:
        self.__post_init__()
        if type(contract) is not FiniteVariationContract:
            raise TypeError("contract must be an exact FiniteVariationContract")
        contract.__post_init__()
        candidate_pool_size = (
            len(contract.options)
            if self.candidate_pool_size is None
            else self.candidate_pool_size
        )
        _validate_inputs(
            replicate_seed=self.replicate_seed,
            benchmark_sha256=benchmark_sha256,
            wave_index=wave_index,
            parent_configuration_sha256=parent_configuration_sha256,
            contract=contract,
            candidate_pool_size=candidate_pool_size,
            model_selection_size=self.model_selection_size,
            evaluation_size=evaluation_size,
            min_distinct_families=min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=required_option_ids,
        )
        state = _state_record(
            replicate_seed=self.replicate_seed,
            benchmark_sha256=benchmark_sha256,
            wave_index=wave_index,
            parent_configuration_sha256=parent_configuration_sha256,
            finite_contract_sha256=contract.identity_sha256,
            evaluation_size=evaluation_size,
            min_distinct_families=min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=required_option_ids,
        )
        state_identity = _hash(_STATE_DOMAIN, state)
        all_option_ids = tuple(option.option_id for option in contract.options)
        keyed_order = tuple(
            sorted(
                all_option_ids,
                key=lambda option_id: (
                    hashlib.sha256(
                        _OPTION_ORDER_DOMAIN
                        + bytes.fromhex(state_identity)
                        + option_id.encode("ascii", errors="strict")
                    ).digest(),
                    option_id,
                ),
            )
        )
        if require_pairwise_disjoint_parent_patches:
            witness = pairwise_disjoint_parent_patch_witness(
                contract,
                all_option_ids,
                portfolio_size=evaluation_size,
                min_distinct_families=min_distinct_families,
                ordering_key_sha256=state_identity,
            )
            if witness is None:  # The public request should already fail first.
                raise ValueError("finite contract has no common-pool witness")
        else:
            witness = keyed_order[:evaluation_size]
        members = list(witness)
        members.extend(
            option_id for option_id in required_option_ids if option_id not in members
        )
        if len(members) > candidate_pool_size:
            raise ValueError(
                "candidate pool cannot contain both its feasibility witness and "
                "all required options"
            )
        members.extend(
            option_id
            for option_id in keyed_order
            if option_id not in members
        )
        selected_members = frozenset(members[:candidate_pool_size])
        presentation_order = tuple(
            sorted(
                all_option_ids,
                key=lambda option_id: (
                    hashlib.sha256(
                        _PRESENTATION_ORDER_DOMAIN
                        + bytes.fromhex(state_identity)
                        + option_id.encode("ascii", errors="strict")
                    ).digest(),
                    option_id,
                ),
            )
        )
        ordered_members = [
            option_id
            for option_id in presentation_order
            if option_id in selected_members
        ]
        # Every pool larger than the hidden feasibility certificate contains
        # this same state-keyed non-certificate guard.  Placing it first keeps
        # the certificate from becoming a visible answer prefix while
        # preserving one relative presentation order for nested M assays.
        witness_set = set(witness)
        required_set = set(required_option_ids)
        presentation_guard = next(
            (
                value
                for value in members
                if value not in witness_set and value not in required_set
            ),
            None,
        )
        if presentation_guard in selected_members:
            ordered_members = [
                presentation_guard,
                *(value for value in ordered_members if value != presentation_guard),
            ]
        option_ids = tuple(ordered_members)
        return CommonCandidatePoolDecision(
            replicate_seed=self.replicate_seed,
            benchmark_sha256=benchmark_sha256,
            wave_index=wave_index,
            parent_configuration_sha256=parent_configuration_sha256,
            finite_contract_sha256=contract.identity_sha256,
            candidate_pool_size=candidate_pool_size,
            model_selection_size=self.model_selection_size,
            evaluation_size=evaluation_size,
            min_distinct_families=min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=required_option_ids,
            option_ids=option_ids,
            feasibility_witness_option_ids=tuple(witness),
        )

    def require_decision(
        self,
        decision: CommonCandidatePoolDecision,
        *,
        benchmark_sha256: str,
        wave_index: int,
        parent_configuration_sha256: str,
        contract: FiniteVariationContract,
        evaluation_size: int,
        min_distinct_families: int | None,
        require_pairwise_disjoint_parent_patches: bool,
        required_option_ids: tuple[str, ...] = (),
    ) -> None:
        if type(decision) is not CommonCandidatePoolDecision:
            raise TypeError("decision must be an exact CommonCandidatePoolDecision")
        expected = self.select(
            benchmark_sha256=benchmark_sha256,
            wave_index=wave_index,
            parent_configuration_sha256=parent_configuration_sha256,
            contract=contract,
            evaluation_size=evaluation_size,
            min_distinct_families=min_distinct_families,
            require_pairwise_disjoint_parent_patches=(
                require_pairwise_disjoint_parent_patches
            ),
            required_option_ids=required_option_ids,
        )
        if decision.to_record() != expected.to_record():
            raise ValueError("common candidate pool differs from exact policy replay")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.policy_definition_sha256,
            "replicate_seed": self.replicate_seed,
            "candidate_pool_size": self.candidate_pool_size,
            "candidate_pool_mode": (
                "complete_finite_contract"
                if self.candidate_pool_size is None
                else "fixed_size"
            ),
            "model_selection_size": self.model_selection_size,
        }


__all__ = [
    "CommonCandidatePoolDecision",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "TaskKeyedCommonCandidatePoolPolicy",
]
