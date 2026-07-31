"""Provider- and workload-neutral contextual allocation contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_CONTRACT_DOMAIN = b"agent-evolve:contextual-portfolio-allocation-contract:v1\x00"
_REALIZATION_DOMAIN = b"agent-evolve:contextual-portfolio-allocation-realization:v1\x00"
_CAPABILITY_WITNESS_DOMAIN = (
    b"agent-evolve:contextual-arm-count-capability-witness:v1\x00"
)
_CAPABILITY_DOMAIN = b"agent-evolve:contextual-arm-count-capability:v1\x00"
_JOINT_VECTOR_DOMAIN = b"agent-evolve:contextual-joint-count-vector:v1\x00"
_LANE_JOINT_CAPABILITY_DOMAIN = (
    b"agent-evolve:contextual-lane-joint-count-capability:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _validate_counts(
    values: tuple[tuple[str, int], ...],
    *,
    name: str,
    expected_slots: int,
) -> None:
    if type(values) is not tuple or not values:
        raise ValueError(f"{name} must be a non-empty exact tuple")
    for arm_id, count in values:
        _require_token(arm_id, name=f"{name}.arm_id")
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} counts must be non-negative")
    if values != tuple(sorted(values)) or len({value[0] for value in values}) != len(
        values
    ):
        raise ValueError(f"{name} must use canonical unique arms")
    if sum(value[1] for value in values) != expected_slots:
        raise ValueError(f"{name} must allocate every evaluation slot")


@dataclass(frozen=True, slots=True)
class ContextualArmCountCapabilityWitness:
    """One prior, actually realized stage-level arm-count vector.

    A witness says only that an allocation was realizable under an earlier
    prior-wave candidate set.  It is deliberately weaker than a promise about
    the current wave.  Keeping that distinction explicit lets a controller use
    the evidence conservatively without smuggling workload rules into the
    application layer.
    """

    controller_wave_index: int
    evaluation_slots: int
    realized_target_counts: tuple[tuple[str, int], ...]
    allocation_realization_sha256s: tuple[str, ...]
    witness_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.controller_wave_index) is not int
            or self.controller_wave_index <= 0
        ):
            raise ValueError("controller_wave_index must be positive")
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        _validate_counts(
            self.realized_target_counts,
            name="realized_target_counts",
            expected_slots=self.evaluation_slots,
        )
        if (
            type(self.allocation_realization_sha256s) is not tuple
            or not self.allocation_realization_sha256s
            or self.allocation_realization_sha256s
            != tuple(sorted(set(self.allocation_realization_sha256s)))
        ):
            raise ValueError(
                "allocation realization hashes must be non-empty and canonical"
            )
        for value in self.allocation_realization_sha256s:
            require_sha256(value, "allocation_realization_sha256")
        object.__setattr__(
            self,
            "witness_sha256",
            hashlib.sha256(
                _CAPABILITY_WITNESS_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "controller_wave_index": self.controller_wave_index,
            "evaluation_slots": self.evaluation_slots,
            "realized_target_counts": [
                list(value) for value in self.realized_target_counts
            ],
            "allocation_realization_sha256s": list(self.allocation_realization_sha256s),
            "interpretation": "prior_realized_witness_not_current_capacity_promise",
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "witness_sha256": self.witness_sha256}


@dataclass(frozen=True, slots=True)
class ContextualArmCountCapability:
    """A workload-blind empirical feasible-count set for one arm kind."""

    kind: str
    evaluation_slots: int
    arm_ids: tuple[str, ...]
    witnesses: tuple[ContextualArmCountCapabilityWitness, ...]
    capability_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in {"source", "operator"}:
            raise ValueError("kind must be source or operator")
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        if type(self.arm_ids) is not tuple or not self.arm_ids:
            raise ValueError("arm_ids must be a non-empty exact tuple")
        for value in self.arm_ids:
            _require_token(value, name="arm_id")
        if self.arm_ids != tuple(sorted(set(self.arm_ids))):
            raise ValueError("arm_ids must be unique and canonical")
        if (
            type(self.witnesses) is not tuple
            or not self.witnesses
            or any(
                type(value) is not ContextualArmCountCapabilityWitness
                for value in self.witnesses
            )
        ):
            raise ValueError("witnesses must contain exact capability witnesses")
        for value in self.witnesses:
            value.__post_init__()
            if value.evaluation_slots != self.evaluation_slots:
                raise ValueError("capability witness has a different stage width")
            if tuple(arm_id for arm_id, _ in value.realized_target_counts) != (
                self.arm_ids
            ):
                raise ValueError("capability witness has different arms")
        if tuple(value.witness_sha256 for value in self.witnesses) != tuple(
            sorted({value.witness_sha256 for value in self.witnesses})
        ):
            raise ValueError("capability witnesses must be unique and canonical")
        object.__setattr__(
            self,
            "capability_sha256",
            hashlib.sha256(
                _CAPABILITY_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def feasible_count_vectors(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        """Return unique witnessed vectors in deterministic count order."""

        return tuple(sorted({value.realized_target_counts for value in self.witnesses}))

    @property
    def allocation_realization_sha256s(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    receipt
                    for witness in self.witnesses
                    for receipt in witness.allocation_realization_sha256s
                }
            )
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "evaluation_slots": self.evaluation_slots,
            "arm_ids": list(self.arm_ids),
            "witnesses": [value.to_record() for value in self.witnesses],
            "feasible_count_vectors": [
                [list(value) for value in vector]
                for vector in self.feasible_count_vectors
            ],
            "current_capacity_guaranteed": False,
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "capability_sha256": self.capability_sha256,
        }


@dataclass(frozen=True, slots=True)
class ContextualJointCountVector:
    """One prospectively witnessed source x operator marginal pair.

    The vector deliberately carries only arm counts and immutable finite-option
    identities.  Objective values, ranks, model outputs, and workload names are
    outside this port.  A lane capability may retain one canonical witness for
    each distinct marginal pair even when many option subsets realize it.
    """

    source_target_counts: tuple[tuple[str, int], ...]
    operator_target_counts: tuple[tuple[str, int], ...]
    feasibility_witness_option_identity_sha256s: tuple[str, ...]
    vector_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        slots = sum(value[1] for value in self.source_target_counts)
        if slots <= 0:
            raise ValueError("joint count vector must cover positive capacity")
        _validate_counts(
            self.source_target_counts,
            name="source_target_counts",
            expected_slots=slots,
        )
        _validate_counts(
            self.operator_target_counts,
            name="operator_target_counts",
            expected_slots=slots,
        )
        witnesses = self.feasibility_witness_option_identity_sha256s
        if (
            type(witnesses) is not tuple
            or len(witnesses) != slots
            or witnesses != tuple(sorted(set(witnesses)))
        ):
            raise ValueError(
                "joint count vector requires one canonical unique witness per slot"
            )
        for value in witnesses:
            require_sha256(value, "feasibility_witness_option_identity_sha256")
        object.__setattr__(
            self,
            "vector_sha256",
            hashlib.sha256(
                _JOINT_VECTOR_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @property
    def evaluation_slots(self) -> int:
        return sum(value[1] for value in self.source_target_counts)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_target_counts": [
                list(value) for value in self.source_target_counts
            ],
            "operator_target_counts": [
                list(value) for value in self.operator_target_counts
            ],
            "feasibility_witness_option_identity_sha256s": list(
                self.feasibility_witness_option_identity_sha256s
            ),
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "vector_sha256": self.vector_sha256}


@dataclass(frozen=True, slots=True)
class ContextualLaneJointCountCapability:
    """Exact finite-contract marginal capability for one concurrent lane."""

    slice_id: str
    finite_contract_identity_sha256: str
    structural_constraint_sha256: str
    evaluation_slots: int
    source_arm_ids: tuple[str, ...]
    operator_arm_ids: tuple[str, ...]
    feasible_vectors: tuple[ContextualJointCountVector, ...]
    minimum_single_path_interventions: int = 0
    minimum_disjoint_parent_patch_pairs: int = 0
    capability_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.slice_id, name="slice_id")
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        require_sha256(
            self.structural_constraint_sha256,
            "structural_constraint_sha256",
        )
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        if (
            type(self.minimum_single_path_interventions) is not int
            or not 0
            <= self.minimum_single_path_interventions
            <= self.evaluation_slots
        ):
            raise ValueError(
                "minimum_single_path_interventions must lie in lane capacity"
            )
        maximum_pairs = self.evaluation_slots * (self.evaluation_slots - 1) // 2
        if (
            type(self.minimum_disjoint_parent_patch_pairs) is not int
            or not 0
            <= self.minimum_disjoint_parent_patch_pairs
            <= maximum_pairs
        ):
            raise ValueError(
                "minimum_disjoint_parent_patch_pairs must lie in lane pair capacity"
            )
        for name in ("source_arm_ids", "operator_arm_ids"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a non-empty exact tuple")
            for value in values:
                _require_token(value, name=name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if (
            type(self.feasible_vectors) is not tuple
            or not self.feasible_vectors
            or any(
                type(value) is not ContextualJointCountVector
                for value in self.feasible_vectors
            )
        ):
            raise ValueError("feasible_vectors must contain exact joint vectors")
        for value in self.feasible_vectors:
            value.__post_init__()
            if value.evaluation_slots != self.evaluation_slots:
                raise ValueError("joint vector has a different lane capacity")
            if tuple(arm_id for arm_id, _ in value.source_target_counts) != (
                self.source_arm_ids
            ):
                raise ValueError("joint vector has different source arms")
            if tuple(arm_id for arm_id, _ in value.operator_target_counts) != (
                self.operator_arm_ids
            ):
                raise ValueError("joint vector has different operator arms")
        vector_hashes = tuple(value.vector_sha256 for value in self.feasible_vectors)
        if vector_hashes != tuple(sorted(set(vector_hashes))):
            raise ValueError("joint vectors must be unique and hash-canonical")
        marginal_pairs = tuple(
            (value.source_target_counts, value.operator_target_counts)
            for value in self.feasible_vectors
        )
        if len(set(marginal_pairs)) != len(marginal_pairs):
            raise ValueError("joint capability repeats one marginal pair")
        object.__setattr__(
            self,
            "capability_sha256",
            hashlib.sha256(
                _LANE_JOINT_CAPABILITY_DOMAIN
                + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def resolve_vector(self, vector_sha256: str) -> ContextualJointCountVector:
        require_sha256(vector_sha256, "vector_sha256")
        try:
            return next(
                value
                for value in self.feasible_vectors
                if value.vector_sha256 == vector_sha256
            )
        except StopIteration as error:
            raise ValueError(
                "joint vector is absent from this lane capability"
            ) from error

    def _unsigned_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": (
                3
                if self.minimum_disjoint_parent_patch_pairs
                else 2
                if self.minimum_single_path_interventions
                else 1
            ),
            "slice_id": self.slice_id,
            "finite_contract_identity_sha256": self.finite_contract_identity_sha256,
            "structural_constraint_sha256": self.structural_constraint_sha256,
            "evaluation_slots": self.evaluation_slots,
            "source_arm_ids": list(self.source_arm_ids),
            "operator_arm_ids": list(self.operator_arm_ids),
            "feasible_vectors": [value.to_record() for value in self.feasible_vectors],
            "current_capacity_guaranteed": True,
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }
        if self.minimum_single_path_interventions:
            record["minimum_single_path_interventions"] = (
                self.minimum_single_path_interventions
            )
            record["intervention_axis"] = (
                "exact_parent_relative_changed_json_path_count"
            )
        if self.minimum_disjoint_parent_patch_pairs:
            record["minimum_disjoint_parent_patch_pairs"] = (
                self.minimum_disjoint_parent_patch_pairs
            )
            record["offspring_opportunity_axis"] = (
                "pairwise_disjoint_parent_relative_patch_pairs"
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "capability_sha256": self.capability_sha256}


@dataclass(frozen=True, slots=True)
class ContextualPortfolioAllocationContract:
    """Exact request-local exposure targets from a prior-only controller."""

    campaign_scope_sha256: str
    query_sha256: str
    decision_sha256: str
    campaign_generation: int
    controller_wave_index: int
    phase_id: str
    slice_id: str
    evaluation_slots: int
    source_target_counts: tuple[tuple[str, int], ...]
    operator_target_counts: tuple[tuple[str, int], ...]
    minimum_single_path_interventions: int = 0
    minimum_disjoint_parent_patch_pairs: int = 0
    feasibility_witness_option_identity_sha256s: tuple[str, ...] = ()
    contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_scope_sha256",
            "query_sha256",
            "decision_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("campaign_generation", "controller_wave_index"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive")
        _require_token(self.phase_id, name="phase_id")
        _require_token(self.slice_id, name="slice_id")
        if type(self.evaluation_slots) is not int or self.evaluation_slots <= 0:
            raise ValueError("evaluation_slots must be positive")
        if (
            type(self.minimum_single_path_interventions) is not int
            or not 0
            <= self.minimum_single_path_interventions
            <= self.evaluation_slots
        ):
            raise ValueError(
                "minimum_single_path_interventions must lie in allocation capacity"
            )
        maximum_pairs = self.evaluation_slots * (self.evaluation_slots - 1) // 2
        if (
            type(self.minimum_disjoint_parent_patch_pairs) is not int
            or not 0
            <= self.minimum_disjoint_parent_patch_pairs
            <= maximum_pairs
        ):
            raise ValueError(
                "minimum_disjoint_parent_patch_pairs must lie in allocation pair "
                "capacity"
            )
        _validate_counts(
            self.source_target_counts,
            name="source_target_counts",
            expected_slots=self.evaluation_slots,
        )
        _validate_counts(
            self.operator_target_counts,
            name="operator_target_counts",
            expected_slots=self.evaluation_slots,
        )
        witnesses = self.feasibility_witness_option_identity_sha256s
        if type(witnesses) is not tuple:
            raise TypeError(
                "feasibility_witness_option_identity_sha256s must be an exact tuple"
            )
        if witnesses:
            if len(witnesses) != self.evaluation_slots or witnesses != tuple(
                sorted(set(witnesses))
            ):
                raise ValueError(
                    "allocation feasibility witness must contain one canonical "
                    "unique option identity per evaluation slot"
                )
            for value in witnesses:
                require_sha256(value, "feasibility_witness_option_identity_sha256")
        object.__setattr__(
            self,
            "contract_sha256",
            hashlib.sha256(
                _CONTRACT_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": (
                4
                if self.feasibility_witness_option_identity_sha256s
                else 3
                if self.minimum_disjoint_parent_patch_pairs
                else 2
                if self.minimum_single_path_interventions
                else 1
            ),
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "query_sha256": self.query_sha256,
            "decision_sha256": self.decision_sha256,
            "campaign_generation": self.campaign_generation,
            "controller_wave_index": self.controller_wave_index,
            "phase_id": self.phase_id,
            "slice_id": self.slice_id,
            "evaluation_slots": self.evaluation_slots,
            "source_target_counts": [
                list(value) for value in self.source_target_counts
            ],
            "operator_target_counts": [
                list(value) for value in self.operator_target_counts
            ],
        }
        if self.minimum_single_path_interventions:
            record["minimum_single_path_interventions"] = (
                self.minimum_single_path_interventions
            )
            record["intervention_axis"] = (
                "exact_parent_relative_changed_json_path_count"
            )
        if self.minimum_disjoint_parent_patch_pairs:
            record["minimum_disjoint_parent_patch_pairs"] = (
                self.minimum_disjoint_parent_patch_pairs
            )
            record["offspring_opportunity_axis"] = (
                "pairwise_disjoint_parent_relative_patch_pairs"
            )
        if self.feasibility_witness_option_identity_sha256s:
            record["feasibility_witness_option_identity_sha256s"] = list(
                self.feasibility_witness_option_identity_sha256s
            )
            record["feasibility_witness_semantics"] = (
                "current_finite_contract_exact_joint_count_and_structural_witness"
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "contract_sha256": self.contract_sha256}

    def target_count(self, *, kind: str, arm_id: str) -> int:
        if kind not in {"source", "operator"}:
            raise ValueError("kind must be source or operator")
        _require_token(arm_id, name="arm_id")
        values = (
            self.source_target_counts
            if kind == "source"
            else self.operator_target_counts
        )
        try:
            return dict(values)[arm_id]
        except KeyError as error:
            raise ValueError("arm is absent from this allocation contract") from error


@dataclass(frozen=True, slots=True)
class ContextualPortfolioAllocationRealization:
    """Objective-blind requested-to-realized exposure evidence.

    This is an inward-facing application port: an integration may solve a
    workload-specific finite feasibility problem, but the controller receives
    only canonical arm counts and the identity of the prospective contract.
    Candidate fields, objective values, workload IDs, model IDs, and provider
    metadata are deliberately absent.
    """

    campaign_scope_sha256: str
    query_sha256: str
    decision_sha256: str
    contract_sha256: str
    controller_wave_index: int
    slice_id: str
    requested_source_target_counts: tuple[tuple[str, int], ...]
    requested_operator_target_counts: tuple[tuple[str, int], ...]
    realized_source_target_counts: tuple[tuple[str, int], ...]
    realized_operator_target_counts: tuple[tuple[str, int], ...]
    requested_minimum_single_path_interventions: int = 0
    realized_single_path_interventions: int = 0
    requested_minimum_disjoint_parent_patch_pairs: int = 0
    realized_disjoint_parent_patch_pairs: int = 0
    realization_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_scope_sha256",
            "query_sha256",
            "decision_sha256",
            "contract_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if (
            type(self.controller_wave_index) is not int
            or self.controller_wave_index <= 0
        ):
            raise ValueError("controller_wave_index must be positive")
        _require_token(self.slice_id, name="slice_id")
        slots = sum(value[1] for value in self.requested_source_target_counts)
        if slots <= 0:
            raise ValueError("allocation realization must cover positive capacity")
        for name in (
            "requested_source_target_counts",
            "requested_operator_target_counts",
            "realized_source_target_counts",
            "realized_operator_target_counts",
        ):
            _validate_counts(
                getattr(self, name),
                name=name,
                expected_slots=slots,
            )
        if tuple(value[0] for value in self.requested_source_target_counts) != tuple(
            value[0] for value in self.realized_source_target_counts
        ):
            raise ValueError("realized source arms differ from requested arms")
        if tuple(value[0] for value in self.requested_operator_target_counts) != tuple(
            value[0] for value in self.realized_operator_target_counts
        ):
            raise ValueError("realized operator arms differ from requested arms")
        if (
            type(self.requested_minimum_single_path_interventions) is not int
            or not 0
            <= self.requested_minimum_single_path_interventions
            <= slots
        ):
            raise ValueError(
                "requested minimum single-path interventions is invalid"
            )
        if (
            type(self.realized_single_path_interventions) is not int
            or not 0 <= self.realized_single_path_interventions <= slots
        ):
            raise ValueError("realized single-path interventions is invalid")
        if self.realized_single_path_interventions < (
            self.requested_minimum_single_path_interventions
        ):
            raise ValueError("realized portfolio violated its single-path floor")
        maximum_pairs = slots * (slots - 1) // 2
        if (
            type(self.requested_minimum_disjoint_parent_patch_pairs) is not int
            or not 0
            <= self.requested_minimum_disjoint_parent_patch_pairs
            <= maximum_pairs
        ):
            raise ValueError(
                "requested minimum disjoint parent-patch pairs is invalid"
            )
        if (
            type(self.realized_disjoint_parent_patch_pairs) is not int
            or not 0 <= self.realized_disjoint_parent_patch_pairs <= maximum_pairs
        ):
            raise ValueError("realized disjoint parent-patch pair count is invalid")
        if self.realized_disjoint_parent_patch_pairs < (
            self.requested_minimum_disjoint_parent_patch_pairs
        ):
            raise ValueError(
                "realized portfolio violated its disjoint parent-pair floor"
            )
        object.__setattr__(
            self,
            "realization_sha256",
            hashlib.sha256(
                _REALIZATION_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @staticmethod
    def _l1(
        requested: tuple[tuple[str, int], ...],
        realized: tuple[tuple[str, int], ...],
    ) -> int:
        return sum(
            abs(left[1] - right[1])
            for left, right in zip(requested, realized, strict=True)
        )

    @property
    def source_l1_deviation(self) -> int:
        return self._l1(
            self.requested_source_target_counts,
            self.realized_source_target_counts,
        )

    @property
    def operator_l1_deviation(self) -> int:
        return self._l1(
            self.requested_operator_target_counts,
            self.realized_operator_target_counts,
        )

    @property
    def exact(self) -> bool:
        return self.source_l1_deviation == 0 and self.operator_l1_deviation == 0

    def _unsigned_record(self) -> dict[str, object]:
        intervention_bound = self.requested_minimum_single_path_interventions > 0
        offspring_opportunity_bound = (
            self.requested_minimum_disjoint_parent_patch_pairs > 0
        )
        record: dict[str, object] = {
            "schema_version": (
                3 if offspring_opportunity_bound else 2 if intervention_bound else 1
            ),
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "query_sha256": self.query_sha256,
            "decision_sha256": self.decision_sha256,
            "contract_sha256": self.contract_sha256,
            "controller_wave_index": self.controller_wave_index,
            "slice_id": self.slice_id,
            "requested_source_target_counts": [
                list(value) for value in self.requested_source_target_counts
            ],
            "requested_operator_target_counts": [
                list(value) for value in self.requested_operator_target_counts
            ],
            "realized_source_target_counts": [
                list(value) for value in self.realized_source_target_counts
            ],
            "realized_operator_target_counts": [
                list(value) for value in self.realized_operator_target_counts
            ],
            "source_l1_deviation": self.source_l1_deviation,
            "operator_l1_deviation": self.operator_l1_deviation,
            "exact": self.exact,
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }
        if intervention_bound:
            record["requested_minimum_single_path_interventions"] = (
                self.requested_minimum_single_path_interventions
            )
            record["realized_single_path_interventions"] = (
                self.realized_single_path_interventions
            )
            record["intervention_axis"] = (
                "exact_parent_relative_changed_json_path_count"
            )
        if offspring_opportunity_bound:
            record["requested_minimum_disjoint_parent_patch_pairs"] = (
                self.requested_minimum_disjoint_parent_patch_pairs
            )
            record["realized_disjoint_parent_patch_pairs"] = (
                self.realized_disjoint_parent_patch_pairs
            )
            record["offspring_opportunity_axis"] = (
                "pairwise_disjoint_parent_relative_patch_pairs"
            )
        return record

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "realization_sha256": self.realization_sha256,
        }

    def require_contract(
        self,
        contract: ContextualPortfolioAllocationContract,
    ) -> None:
        if type(contract) is not ContextualPortfolioAllocationContract:
            raise TypeError("contract must be exact")
        contract.__post_init__()
        if (
            self.campaign_scope_sha256 != contract.campaign_scope_sha256
            or self.query_sha256 != contract.query_sha256
            or self.decision_sha256 != contract.decision_sha256
            or self.contract_sha256 != contract.contract_sha256
            or self.controller_wave_index != contract.controller_wave_index
            or self.slice_id != contract.slice_id
            or self.requested_source_target_counts != contract.source_target_counts
            or self.requested_operator_target_counts != contract.operator_target_counts
            or self.requested_minimum_single_path_interventions
            != contract.minimum_single_path_interventions
            or self.requested_minimum_disjoint_parent_patch_pairs
            != contract.minimum_disjoint_parent_patch_pairs
        ):
            raise ValueError("allocation realization differs from its contract")


__all__ = [
    "ContextualArmCountCapability",
    "ContextualArmCountCapabilityWitness",
    "ContextualJointCountVector",
    "ContextualLaneJointCountCapability",
    "ContextualPortfolioAllocationContract",
    "ContextualPortfolioAllocationRealization",
]
