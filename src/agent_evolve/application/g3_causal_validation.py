"""Terminal integrity and mechanism decisions for the generic G3 screen.

The G3 planner validates chronology while constructing each next wave, but it
never observes the outcomes of its final zero-call wave.  This module supplies
the independent post-G3 gate.  A feedback interceptor must call
``validate_g3_terminal_state`` before dispatching the optional curation call.
The same authenticated core is re-used after optimization to validate the
six-call terminal result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InsightAssignmentKind,
    InvocationOutcome,
    OperatorKind,
    ProposalAuthority,
    ReflectionCallReceipt,
    ReflectionCallStatus,
    ReflectionPublicationResult,
)
from agent_evolve.application.budgeted_optimizer import (
    GenerationReceipt,
    OptimizerResult,
    OptimizerState,
    validate_generation_receipt_integrity,
    validate_optimizer_result_integrity,
)
from agent_evolve.application.g3_causal_screen import (
    G1_DIAGNOSTIC_SLOT_IDS,
    G2_SLOT_IDS,
    G3_SLOT_IDS,
    G3CausalScreenPlanner,
    G3ExpectedEndpoint,
    G3ExpectedUnion,
    G3TerminalValidationAuthority,
    G3_SCREEN_BUDGET,
)
from agent_evolve.application.g3_postseal_curation import (
    G3PostsealCurationAuthority,
    G3PostsealCurationReceipt,
    G3PostsealCurationSpec,
    build_g3_postseal_curation_reservation,
)
from agent_evolve.application.generation_feedback import (
    validate_generation_feedback_receipt,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import typed_json_equal
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentAssignmentRole,
)


_RECEIPT_DOMAIN = b"agent-evolve:g3-terminal-validation:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:g3-result-validation:v2-curation-bound\x00"
_SEED_OCCURRENCE_DOMAIN = b"agent-evolve:g3-seed-occurrence-binding:v1\x00"
_CACHE_KEYS = (
    "cached_entries",
    "capacity",
    "coalesced",
    "evictions",
    "hits",
    "in_flight",
    "misses",
)
_EXPECTED_CACHE = {
    "capacity": None,
    "cached_entries": 11,
    "in_flight": 0,
    "hits": 1,
    "misses": 11,
    "coalesced": 0,
    "evictions": 0,
}


class G3TerminalValidationError(ValueError):
    """A completed G3 state/result violated its frozen causal protocol."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, record: object) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _fail(message: str) -> None:
    raise G3TerminalValidationError(message)


@dataclass(frozen=True, slots=True)
class G3MechanismDecision:
    """Exact single-block H1--H3 contrasts and preregistered decision."""

    q_parent: float
    q_adaptive: float
    q_score_shuffled: float
    q_neutral: float
    q_engine_mate: float
    q_adaptive_union: float
    q_score_shuffled_union: float
    q_neutral_union: float
    delta_as_direct: float = field(init=False)
    delta_an_direct: float = field(init=False)
    delta_as_union: float = field(init=False)
    delta_an_union: float = field(init=False)
    i_adaptive: float = field(init=False)
    i_score_shuffled: float = field(init=False)
    i_neutral: float = field(init=False)
    j_adaptive: float = field(init=False)
    j_score_shuffled: float = field(init=False)
    j_neutral: float = field(init=False)
    h1_pass: bool = field(init=False)
    h2_pass: bool = field(init=False)
    h3_pass: bool = field(init=False)
    advance_to_replication: bool = field(init=False)
    kill_reasons: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        names = (
            "q_parent",
            "q_adaptive",
            "q_score_shuffled",
            "q_neutral",
            "q_engine_mate",
            "q_adaptive_union",
            "q_score_shuffled_union",
            "q_neutral_union",
        )
        for name in names:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite canonical float")
        delta_as_direct = self.q_adaptive - self.q_score_shuffled
        delta_an_direct = self.q_adaptive - self.q_neutral
        delta_as_union = self.q_adaptive_union - self.q_score_shuffled_union
        delta_an_union = self.q_adaptive_union - self.q_neutral_union
        i_adaptive = self.q_adaptive_union - max(
            self.q_adaptive,
            self.q_engine_mate,
        )
        i_score_shuffled = self.q_score_shuffled_union - max(
            self.q_score_shuffled,
            self.q_engine_mate,
        )
        i_neutral = self.q_neutral_union - max(
            self.q_neutral,
            self.q_engine_mate,
        )
        j_adaptive = (
            self.q_adaptive_union
            - self.q_adaptive
            - self.q_engine_mate
            + self.q_parent
        )
        j_score_shuffled = (
            self.q_score_shuffled_union
            - self.q_score_shuffled
            - self.q_engine_mate
            + self.q_parent
        )
        j_neutral = (
            self.q_neutral_union
            - self.q_neutral
            - self.q_engine_mate
            + self.q_parent
        )
        derived = {
            "delta_as_direct": delta_as_direct,
            "delta_an_direct": delta_an_direct,
            "delta_as_union": delta_as_union,
            "delta_an_union": delta_an_union,
            "i_adaptive": i_adaptive,
            "i_score_shuffled": i_score_shuffled,
            "i_neutral": i_neutral,
            "j_adaptive": j_adaptive,
            "j_score_shuffled": j_score_shuffled,
            "j_neutral": j_neutral,
        }
        if any(not math.isfinite(value) for value in derived.values()):
            raise ValueError("G3 contrast arithmetic produced a non-finite value")
        for name, value in derived.items():
            object.__setattr__(self, name, float(value))
        h1 = delta_as_direct > 0.0 and delta_an_direct > 0.0
        h2 = delta_as_union > 0.0 and delta_an_union > 0.0
        h3 = i_adaptive > 0.0
        reasons = tuple(
            reason
            for passed, reason in (
                (h1, "adaptive_did_not_beat_both_direct_controls"),
                (h2, "adaptive_advantage_did_not_survive_recombination"),
                (h3, "adaptive_union_did_not_beat_both_parents"),
            )
            if not passed
        )
        object.__setattr__(self, "h1_pass", h1)
        object.__setattr__(self, "h2_pass", h2)
        object.__setattr__(self, "h3_pass", h3)
        object.__setattr__(self, "advance_to_replication", h1 and h2 and h3)
        object.__setattr__(self, "kill_reasons", reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "q": {
                "P_H": self.q_parent.hex(),
                "A": self.q_adaptive.hex(),
                "S": self.q_score_shuffled.hex(),
                "N": self.q_neutral.hex(),
                "E": self.q_engine_mate.hex(),
                "A_union_E": self.q_adaptive_union.hex(),
                "S_union_E": self.q_score_shuffled_union.hex(),
                "N_union_E": self.q_neutral_union.hex(),
            },
            "contrasts": {
                "delta_as_direct": self.delta_as_direct.hex(),
                "delta_an_direct": self.delta_an_direct.hex(),
                "delta_as_union": self.delta_as_union.hex(),
                "delta_an_union": self.delta_an_union.hex(),
                "i_adaptive": self.i_adaptive.hex(),
                "i_score_shuffled": self.i_score_shuffled.hex(),
                "i_neutral": self.i_neutral.hex(),
                "j_adaptive": self.j_adaptive.hex(),
                "j_score_shuffled": self.j_score_shuffled.hex(),
                "j_neutral": self.j_neutral.hex(),
            },
            "h1_pass": self.h1_pass,
            "h2_pass": self.h2_pass,
            "h3_pass": self.h3_pass,
            "advance_to_replication": self.advance_to_replication,
            "kill_reasons": list(self.kill_reasons),
        }


@dataclass(frozen=True, slots=True)
class G3TerminalStateValidationReceipt:
    """Authenticated proof that all optimization endpoints sealed correctly."""

    authority_sha256: str
    endpoint_definition_sha256: str
    archive_snapshot_sha256: str
    generation_receipt_sha256s: tuple[str, str, str]
    feedback_receipt_sha256s: tuple[str, str]
    occurrence_ids: tuple[str, ...]
    configuration_sha256s: tuple[str, ...]
    phenotype_identity_sha256s: tuple[str, ...]
    cache_evidence: tuple[tuple[str, int | None], ...]
    mechanism_decision: G3MechanismDecision
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for value in (
            self.authority_sha256,
            self.endpoint_definition_sha256,
            self.archive_snapshot_sha256,
            *self.generation_receipt_sha256s,
            *self.feedback_receipt_sha256s,
            *self.configuration_sha256s,
            *self.phenotype_identity_sha256s,
        ):
            require_sha256(value, "G3 terminal receipt digest")
        if len(self.occurrence_ids) != 12 or len(set(self.occurrence_ids)) != 12:
            raise ValueError("G3 terminal receipt requires 12 unique occurrences")
        if len(self.configuration_sha256s) != 12:
            raise ValueError("G3 terminal receipt requires 12 configurations")
        if len(self.phenotype_identity_sha256s) != 12:
            raise ValueError("G3 terminal receipt requires 12 phenotypes")
        if type(self.mechanism_decision) is not G3MechanismDecision:
            raise TypeError("mechanism_decision must be exact")
        if self.cache_evidence != tuple(
            (key, _EXPECTED_CACHE[key]) for key in _CACHE_KEYS
        ):
            raise ValueError("G3 terminal cache evidence differs from exact policy")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_RECEIPT_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "authority_sha256": self.authority_sha256,
            "endpoint_definition_sha256": self.endpoint_definition_sha256,
            "archive_snapshot_sha256": self.archive_snapshot_sha256,
            "generation_receipt_sha256s": list(
                self.generation_receipt_sha256s
            ),
            "feedback_receipt_sha256s": list(self.feedback_receipt_sha256s),
            "occurrence_ids": list(self.occurrence_ids),
            "configuration_sha256s": list(self.configuration_sha256s),
            "phenotype_identity_sha256s": list(
                self.phenotype_identity_sha256s
            ),
            "cache_evidence": dict(self.cache_evidence),
            "mechanism_decision": self.mechanism_decision.to_record(),
        }


@dataclass(frozen=True, slots=True)
class G3CausalScreenResultValidationReceipt:
    """Authenticated six-call result, including isolated curation status."""

    optimizer_result_sha256: str
    terminal_state_receipt_sha256: str
    curation_feedback_receipt_sha256: str
    curation_authority_sha256: str
    curation_receipt_sha256: str
    reflection_call_receipt_sha256: str
    curation_status: str
    curation_publication_outcome: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for value in (
            self.optimizer_result_sha256,
            self.terminal_state_receipt_sha256,
            self.curation_feedback_receipt_sha256,
            self.curation_authority_sha256,
            self.curation_receipt_sha256,
            self.reflection_call_receipt_sha256,
        ):
            require_sha256(value, "G3 result validation digest")
        if self.curation_status not in {"sealed_complete", "incomplete"}:
            raise ValueError("curation_status must be sealed_complete or incomplete")
        expected_outcomes = (
            {"completed_revision", "completed_abstention"}
            if self.curation_status == "sealed_complete"
            else {"failed"}
        )
        if self.curation_publication_outcome not in expected_outcomes:
            raise ValueError("curation publication outcome differs from status")
        object.__setattr__(
            self,
            "receipt_sha256",
            _hash(_RESULT_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "optimizer_result_sha256": self.optimizer_result_sha256,
            "terminal_state_receipt_sha256": (
                self.terminal_state_receipt_sha256
            ),
            "curation_feedback_receipt_sha256": (
                self.curation_feedback_receipt_sha256
            ),
            "curation_authority_sha256": self.curation_authority_sha256,
            "curation_receipt_sha256": self.curation_receipt_sha256,
            "reflection_call_receipt_sha256": (
                self.reflection_call_receipt_sha256
            ),
            "curation_status": self.curation_status,
            "curation_publication_outcome": (
                self.curation_publication_outcome
            ),
        }


def _cache_record(
    snapshot: Mapping[str, int | None],
) -> tuple[tuple[str, int | None], ...]:
    if not isinstance(snapshot, Mapping):
        raise TypeError("evaluation_cache_snapshot must be a mapping")
    if set(snapshot) != set(_EXPECTED_CACHE):
        _fail("evaluation cache snapshot has missing or foreign fields")
    observed = {key: snapshot[key] for key in _EXPECTED_CACHE}
    if observed != _EXPECTED_CACHE:
        _fail("evaluation cache is not exact 11 MISS / reproduction-only HIT")
    return tuple((key, observed[key]) for key in _CACHE_KEYS)


def _phenotype_sha256(
    planner: G3CausalScreenPlanner,
    candidate: EvolutionCandidate,
) -> str:
    observed = planner.engine.identify_phenotype(candidate)
    detailed = candidate.detailed_evaluation
    if detailed is not None:
        if not detailed.success or detailed.phenotype != observed:
            _fail("candidate detailed evaluation has inconsistent phenotype evidence")
    return observed.identity_sha256


def _require_success(
    outcome: InvocationOutcome,
    *,
    slot_authority: ProposalAuthority,
    generation: int,
) -> EvolutionCandidate:
    if outcome.failure_stage is not None or outcome.candidate is None:
        _fail("G3 terminal gate observed a failed invocation")
    if outcome.prepared.proposal_authority is not slot_authority:
        _fail("prepared proposal authority differs from its frozen slot")
    if (outcome.prepared.call_id is not None) != (
        slot_authority is ProposalAuthority.MODEL
    ):
        _fail("logical model-call identity differs from proposal authority")
    candidate = outcome.candidate
    if (
        candidate.generation != generation
        or not candidate.valid
        or not candidate.operator_compliant
        or not candidate.evidence_compliant
    ):
        _fail("G3 terminal candidate is invalid or noncompliant")
    if candidate.occurrence.operator_invocation_id != (
        outcome.prepared.operator_invocation_id
    ):
        _fail("candidate occurrence differs from its prepared invocation")
    plan = outcome.prepared.plan
    expected_parent_ids = tuple(parent.candidate_id for parent in plan.parents)
    expected_ancestor_id = (
        None if plan.common_ancestor is None else plan.common_ancestor.candidate_id
    )
    if (
        candidate.operator_kind is not plan.operator_kind
        or candidate.parent_ids != expected_parent_ids
        or candidate.common_ancestor_id != expected_ancestor_id
        or (candidate.call_telemetry is not None)
        != (slot_authority is ProposalAuthority.MODEL)
    ):
        _fail("candidate operator/lineage/telemetry differs from prepared authority")
    return candidate


def _require_expected_endpoint(
    *,
    planner: G3CausalScreenPlanner,
    receipt: GenerationReceipt,
    result_index: int,
    expected: G3ExpectedEndpoint,
    assignment_role: TreatmentAssignmentRole | None,
) -> EvolutionCandidate:
    slot_result = receipt.slot_results[result_index]
    slot = slot_result.slot
    if slot.slot_id != expected.slot_id:
        _fail("endpoint authority differs from receipt slot order")
    candidate = _require_success(
        slot_result.outcome,
        slot_authority=slot.proposal_authority,
        generation=receipt.generation,
    )
    outcome = slot_result.outcome
    if (
        candidate.occurrence.configuration_hash != expected.configuration_sha256
        or not typed_json_equal(candidate.configuration, expected.configuration)
        or _phenotype_sha256(planner, candidate)
        != expected.phenotype_identity_sha256
    ):
        _fail("actual endpoint differs from prospectively frozen endpoint")

    if expected.reference is None:
        if slot.proposal_authority is not ProposalAuthority.ENGINE:
            _fail("reference-free endpoint is not engine-authored")
        if outcome.treatment_admission_receipt is not None:
            _fail("engine endpoint acquired a model treatment receipt")
        return candidate

    if slot.proposal_authority is not ProposalAuthority.MODEL:
        _fail("hypothesis endpoint is not model-authored")
    requirement = slot.plan.insight_treatment_requirement
    preflight = outcome.prepared.treatment_preflight_receipt
    admission = outcome.treatment_admission_receipt
    if (
        requirement is None
        or assignment_role is None
        or requirement.assignment_role is not assignment_role
        or len(requirement.allowed_actions) != 1
        or preflight is None
        or not preflight.passed
        or len(preflight.compatible_actions) != 1
        or admission is None
        or not admission.passed
    ):
        _fail("model endpoint lacks exact successful treatment administration")
    action = requirement.allowed_actions[0]
    if (
        action.option_id != expected.option_id
        or action.option_identity_sha256 != expected.option_identity_sha256
        or preflight.compatible_actions[0].binding() != action
        or admission.selected_action.binding() != action
    ):
        _fail("administered action differs from frozen endpoint action")
    expected_kind = (
        InsightAssignmentKind.QUARANTINE_TEST
        if assignment_role is TreatmentAssignmentRole.SHAM_CONTROL
        else InsightAssignmentKind.RESOLVED_CAUSAL
    )
    if (
        candidate.selected_insight_refs != (expected.reference,)
        or candidate.claimed_insight_ids
        != (expected.reference.insight_id.value,)
        or candidate.insight_assignment_kind is not expected_kind
    ):
        _fail("model endpoint did not instantiate its assigned exact insight")
    return candidate


def _require_absolute_q(
    planner: G3CausalScreenPlanner,
    outcome: InvocationOutcome,
) -> None:
    candidate = outcome.candidate
    assert candidate is not None
    observed = planner.reward_binding.score(
        candidate,
        (),
        planner.engine.objectives,
    )
    if type(observed) is not float or not math.isfinite(observed):
        _fail("absolute endpoint returned a non-finite non-canonical score")
    if observed != outcome.reward:
        _fail("runtime reward changes with operator parents; Q is not absolute")


def validate_g3_terminal_state(
    *,
    state: OptimizerState,
    planner: G3CausalScreenPlanner,
    evaluation_cache_snapshot: Mapping[str, int | None],
) -> G3TerminalStateValidationReceipt:
    """Validate sealed G0--G3 endpoints before any curation provider call."""

    if type(state) is not OptimizerState:
        raise TypeError("state must be an exact OptimizerState")
    OptimizerState.__post_init__(state)
    if type(planner) is not G3CausalScreenPlanner:
        raise TypeError("planner must be an exact G3CausalScreenPlanner")
    authority = planner.terminal_validation_authority
    if type(authority) is not G3TerminalValidationAuthority:
        _fail("planner has no frozen terminal validation authority")
    G3TerminalValidationAuthority.__post_init__(authority)
    if (
        state.generation != 3
        or len(state.candidates) != 12
        or state.unique_evaluations != 11
        or state.logical_llm_calls != 5
        or len(state.generation_receipts) != 3
        or len(state.feedback_receipts) != 2
    ):
        _fail("pre-curation state differs from exact G3 12/11/5 protocol")

    receipts = state.generation_receipts
    for receipt in receipts:
        validate_generation_receipt_integrity(receipt)
    expected_slots = (
        G1_DIAGNOSTIC_SLOT_IDS,
        G2_SLOT_IDS,
        G3_SLOT_IDS,
    )
    if tuple(
        tuple(value.slot.slot_id for value in receipt.slot_results)
        for receipt in receipts
    ) != expected_slots:
        _fail("generation receipt slot order differs from frozen G3 protocol")
    counter_records = tuple(
        (
            receipt.logical_llm_calls_before,
            receipt.logical_llm_calls_after,
            receipt.unique_evaluations_before,
            receipt.unique_evaluations_after,
            receipt.reserved_logical_llm_calls,
            receipt.reserved_unique_evaluations,
        )
        for receipt in receipts
    )
    if counter_records != (
        (0, 2, 2, 4, 2, 2),
        (2, 5, 4, 8, 3, 4),
        (5, 5, 8, 11, 0, 3),
    ):
        _fail("generation counters differ from exact G1/G2/G3 reservations")
    for index, feedback in enumerate(state.feedback_receipts, start=1):
        validate_generation_feedback_receipt(feedback)
        expected_calls = (2, 5)[index - 1]
        if (
            feedback.generation != index
            or feedback.generation_receipt_hash != receipts[index - 1].receipt_hash
            or feedback.reserved_logical_llm_calls != 0
            or feedback.used_logical_llm_calls != 0
            or feedback.logical_llm_calls_before != expected_calls
            or feedback.logical_llm_calls_after != expected_calls
        ):
            _fail("G1/G2 feedback is not an exact zero-call sealed no-op")

    p_d, p_h = state.candidates[:2]
    if any(
        not value.valid
        or not value.operator_compliant
        or not value.evidence_compliant
        for value in (p_d, p_h)
    ):
        _fail("G3 seed state contains an invalid or noncompliant parent")
    if any(
        value.operator_kind is not None
        or value.parent_ids
        or value.common_ancestor_id is not None
        or value.call_telemetry is not None
        for value in (p_d, p_h)
    ):
        _fail("G3 seeds acquired generated-candidate lineage or model telemetry")
    seed_occurrence_record = [
        {
            "candidate_id": value.candidate_id.value,
            "configuration_hash": value.occurrence.configuration_hash,
            "configuration_artifact_hash": (
                value.occurrence.configuration_artifact_hash
            ),
            "proposal_sequence": value.occurrence.proposal_sequence,
            "operator_invocation_id": (
                None
                if value.occurrence.operator_invocation_id is None
                else value.occurrence.operator_invocation_id.value
            ),
        }
        for value in (p_d, p_h)
    ]
    if _hash(_SEED_OCCURRENCE_DOMAIN, seed_occurrence_record) != (
        authority.seed_occurrence_binding_sha256
    ):
        _fail("seed occurrences differ from the exact pre-G1 binding")
    if (
        p_h.candidate_id != authority.hypothesis_parent_candidate_id
        or p_h.occurrence.configuration_hash
        != authority.hypothesis_parent_configuration_sha256
        or not typed_json_equal(
            p_h.configuration,
            authority.hypothesis_parent_configuration,
        )
        or _phenotype_sha256(planner, p_h)
        != authority.hypothesis_parent_phenotype_identity_sha256
    ):
        _fail("held-out parent differs from frozen terminal authority")
    if (
        _phenotype_sha256(planner, p_d),
        _phenotype_sha256(planner, p_h),
    ) != authority.seed_phenotype_identity_sha256s:
        _fail("seed semantic phenotypes differ from frozen authority")

    generated: list[EvolutionCandidate] = []
    for index, expected in enumerate(authority.g1_expected_endpoints):
        generated.append(
            _require_expected_endpoint(
                planner=planner,
                receipt=receipts[0],
                result_index=index,
                expected=expected,
                assignment_role=TreatmentAssignmentRole.ACTIVE,
            )
        )
    for index, expected in enumerate(authority.g2_expected_endpoints):
        generated.append(
            _require_expected_endpoint(
                planner=planner,
                receipt=receipts[1],
                result_index=index,
                expected=expected,
                assignment_role=(
                    TreatmentAssignmentRole.ACTIVE
                    if index < 2
                    else (
                        TreatmentAssignmentRole.SHAM_CONTROL
                        if index == 2
                        else None
                    )
                ),
            )
        )

    reproduction_result = receipts[2].slot_results[0]
    reproduction = _require_success(
        reproduction_result.outcome,
        slot_authority=ProposalAuthority.REPRODUCTION,
        generation=3,
    )
    if (
        reproduction_result.slot.plan.operator_kind is not OperatorKind.REPRODUCTION
        or reproduction_result.slot.plan.parents != (p_h,)
        or reproduction.occurrence.configuration_hash
        != authority.hypothesis_parent_configuration_sha256
        or not typed_json_equal(
            reproduction.configuration,
            authority.hypothesis_parent_configuration,
        )
        or _phenotype_sha256(planner, reproduction)
        != authority.hypothesis_parent_phenotype_identity_sha256
    ):
        _fail("G3 reproduction is not an exact semantic P_H replay")
    generated.append(reproduction)

    for index, expected in enumerate(authority.g3_expected_unions, start=1):
        if type(expected) is not G3ExpectedUnion:
            raise TypeError("union authority must be exact")
        slot_result = receipts[2].slot_results[index]
        slot = slot_result.slot
        union = _require_success(
            slot_result.outcome,
            slot_authority=ProposalAuthority.ENGINE,
            generation=3,
        )
        if (
            slot.slot_id != expected.slot_id
            or slot.plan.operator_kind is not OperatorKind.THREE_WAY_RECOMBINATION
            or slot.plan.common_ancestor != p_h
            or slot.materialized is None
            or slot.materialized.materialization_receipt_hash
            != expected.runtime_materialization_receipt_sha256
            or slot_result.outcome.prepared.materialization_receipt_hash
            != expected.runtime_materialization_receipt_sha256
            or union.preservation_verified is not True
            or union.common_ancestor_id != p_h.candidate_id
            or union.occurrence.configuration_hash != expected.configuration_sha256
            or not typed_json_equal(union.configuration, expected.configuration)
            or _phenotype_sha256(planner, union)
            != expected.phenotype_identity_sha256
        ):
            _fail("actual G3 union differs from prospective/runtime authority")
        generated.append(union)

    if tuple(state.candidates[2:]) != tuple(generated):
        _fail("optimizer candidate history differs from sealed slot outcome order")
    all_outcomes = tuple(
        result.outcome for receipt in receipts for result in receipt.slot_results
    )
    if sum(value.prepared.call_id is not None for value in all_outcomes) != 5:
        _fail("optimization waves did not contain exactly five logical calls")
    for outcome in all_outcomes:
        _require_absolute_q(planner, outcome)

    phenotypes = tuple(
        _phenotype_sha256(planner, candidate) for candidate in state.candidates
    )
    expected_phenotypes = (
        *authority.seed_phenotype_identity_sha256s,
        *(value.phenotype_identity_sha256 for value in authority.g1_expected_endpoints),
        *(value.phenotype_identity_sha256 for value in authority.g2_expected_endpoints),
        authority.hypothesis_parent_phenotype_identity_sha256,
        *(value.phenotype_identity_sha256 for value in authority.g3_expected_unions),
    )
    if phenotypes != expected_phenotypes:
        _fail("terminal phenotype sequence differs from prospective authority")
    if len(set(phenotypes)) != 11 or phenotypes.count(
        authority.hypothesis_parent_phenotype_identity_sha256
    ) != 2:
        _fail("terminal phenotypes do not prove exactly one P_H reproduction reuse")

    g2_rewards = tuple(value.outcome.reward for value in receipts[1].slot_results)
    g3_rewards = tuple(value.outcome.reward for value in receipts[2].slot_results)
    decision = G3MechanismDecision(
        q_parent=g3_rewards[0],
        q_adaptive=g2_rewards[0],
        q_score_shuffled=g2_rewards[1],
        q_neutral=g2_rewards[2],
        q_engine_mate=g2_rewards[3],
        q_adaptive_union=g3_rewards[1],
        q_score_shuffled_union=g3_rewards[2],
        q_neutral_union=g3_rewards[3],
    )
    cache_record = _cache_record(evaluation_cache_snapshot)
    return G3TerminalStateValidationReceipt(
        authority_sha256=authority.authority_sha256,
        endpoint_definition_sha256=planner.endpoint_definition_sha256,
        archive_snapshot_sha256=state.archive_snapshot_hash,
        generation_receipt_sha256s=tuple(
            value.receipt_hash for value in receipts
        ),
        feedback_receipt_sha256s=tuple(
            value.receipt_hash for value in state.feedback_receipts
        ),
        occurrence_ids=tuple(
            value.candidate_id.value for value in state.candidates
        ),
        configuration_sha256s=tuple(
            value.occurrence.configuration_hash for value in state.candidates
        ),
        phenotype_identity_sha256s=phenotypes,
        cache_evidence=cache_record,
        mechanism_decision=decision,
    )


def validate_g3_causal_screen_result(
    result: OptimizerResult,
    *,
    planner: G3CausalScreenPlanner,
    evaluation_cache_snapshot: Mapping[str, int | None],
    curation_spec: G3PostsealCurationSpec,
    curation_authority: G3PostsealCurationAuthority,
    curation_receipt: G3PostsealCurationReceipt,
) -> G3CausalScreenResultValidationReceipt:
    """Validate six-call output against external policy and engine evidence.

    A hash-consistent feedback receipt is insufficient: an attacker can reseal
    arbitrary policy metadata.  This gate independently reconstructs the exact
    reservation/authority from the executed planner and then joins the final
    feedback to the receipt stored by the engine for the actual provider call.
    """

    validate_optimizer_result_integrity(result)
    if type(curation_spec) is not G3PostsealCurationSpec:
        raise TypeError("curation_spec must be exact")
    G3PostsealCurationSpec.__post_init__(curation_spec)
    if type(curation_authority) is not G3PostsealCurationAuthority:
        raise TypeError("curation_authority must be exact")
    G3PostsealCurationAuthority.__post_init__(curation_authority)
    if type(curation_receipt) is not G3PostsealCurationReceipt:
        raise TypeError("curation_receipt must be exact")
    G3PostsealCurationReceipt.__post_init__(curation_receipt)
    state = result.final_state
    if (
        result.budget != G3_SCREEN_BUDGET
        or state.generation != 3
        or state.unique_evaluations != 11
        or state.logical_llm_calls != 6
        or len(state.feedback_receipts) != 3
    ):
        _fail("final optimizer result differs from exact six-call G3 budget")
    pre_curation_state = replace(
        state,
        logical_llm_calls=5,
        feedback_receipts=state.feedback_receipts[:2],
    )
    terminal = validate_g3_terminal_state(
        state=pre_curation_state,
        planner=planner,
        evaluation_cache_snapshot=evaluation_cache_snapshot,
    )
    for generation, feedback in enumerate(state.feedback_receipts[:2], start=1):
        expected_no_op = build_g3_postseal_curation_reservation(
            spec=curation_spec,
            planner=planner,
            memory=planner.memory,
            generation=generation,
        )
        if (
            feedback.policy_id != curation_spec.policy_id
            or feedback.policy_version != curation_spec.policy_version
            or feedback.reservation_hash != expected_no_op.reservation_hash
            or feedback.result_metadata
            != tuple(
                sorted(
                    (
                        ("curation_spec_sha256", curation_spec.spec_sha256),
                        ("curation_status", "not_due"),
                    )
                )
            )
        ):
            _fail("G1/G2 feedback differs from the expected curation policy")

    feedback = state.feedback_receipts[2]
    validate_generation_feedback_receipt(feedback)
    expected_reservation = build_g3_postseal_curation_reservation(
        spec=curation_spec,
        planner=planner,
        memory=planner.memory,
        generation=3,
    )
    receipts = state.generation_receipts
    selected_outcomes = curation_spec.source_scope.select(receipts)
    selected_operator_ids = tuple(
        outcome.prepared.operator_invocation_id
        for outcome in selected_outcomes
    )
    assignments = planner.g2_assignments
    if len(assignments) != 2 or len(
        assignments[0].selection_decision.selected
    ) != 1:
        _fail("final curation has no exact adaptive revision predecessor")
    predecessor = assignments[0].selection_decision.selected[0]
    predecessor_entry = planner.memory.entries_for((predecessor,))[0]
    terminal_authority = planner.terminal_validation_authority
    if terminal_authority is None:
        _fail("final curation lost the terminal validation authority")
    expected_authority = G3PostsealCurationAuthority(
        spec_sha256=curation_spec.spec_sha256,
        reservation_hash=expected_reservation.reservation_hash,
        terminal_validation_receipt_sha256=terminal.receipt_sha256,
        terminal_validation_authority_sha256=(
            terminal_authority.authority_sha256
        ),
        generation_receipt_sha256s=tuple(
            receipt.receipt_hash for receipt in receipts
        ),
        source_scope_sha256=curation_spec.source_scope.scope_sha256,
        source_slot_ids=curation_spec.source_scope.slot_ids,
        source_operator_invocation_ids=selected_operator_ids,
        revision_predecessor=predecessor,
        revision_predecessor_content_sha256=(
            predecessor_entry.draft.content_sha256
        ),
        insight_contract_sha256=(
            curation_spec.insight_contract.identity_sha256
        ),
        reflection_label=curation_spec.label,
    )
    if (
        curation_authority != expected_authority
        or curation_receipt.authority != expected_authority
    ):
        _fail("post-G3 curation authority differs from executed G1--G3 evidence")

    try:
        engine_call_receipt = planner.engine.reflection_call_receipt(
            curation_receipt.call_receipt.call_id
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise G3TerminalValidationError(
            "post-G3 curation has no engine-issued reflection receipt"
        ) from exc
    if type(engine_call_receipt) is not ReflectionCallReceipt:
        _fail("engine returned a foreign reflection receipt type")
    ReflectionCallReceipt.__post_init__(engine_call_receipt)
    if engine_call_receipt != curation_receipt.call_receipt:
        _fail("curation receipt differs from the engine-stored provider call")
    request = engine_call_receipt.request
    if (
        request.source_receipt_sha256s
        != expected_authority.generation_receipt_sha256s
        or request.source_operator_invocation_ids != selected_operator_ids
        or len(request.source_outcome_sha256s) != len(selected_outcomes)
    ):
        _fail("engine call differs from the declared curation evidence scope")

    if engine_call_receipt.status is ReflectionCallStatus.COMPLETED:
        try:
            published_entries = planner.memory.entries_for(
                engine_call_receipt.published_references
            )
            ReflectionPublicationResult(
                entries=published_entries,
                receipt=engine_call_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise G3TerminalValidationError(
                "curation publications differ from current memory/lineage"
            ) from exc
        for entry in published_entries:
            lineage = entry.evidence_lineage
            if (
                lineage is None
                or lineage.available_contrast_ids
                != request.available_contrast_ids
                or not set(lineage.source_operator_invocation_ids).issubset(
                    selected_operator_ids
                )
            ):
                _fail("curation publication has foreign source/call lineage")

    expected_metadata = [
        (
            "curated_entry_count",
            str(len(engine_call_receipt.publications)),
        ),
        ("curation_authority_sha256", expected_authority.authority_sha256),
        ("curation_publication_outcome", curation_receipt.publication_outcome),
        ("curation_receipt_sha256", curation_receipt.receipt_sha256),
        ("curation_spec_sha256", curation_spec.spec_sha256),
        ("curation_status", curation_receipt.curation_status),
        ("reflection_call_id", engine_call_receipt.call_id.value),
        (
            "reflection_call_receipt_sha256",
            engine_call_receipt.receipt_sha256,
        ),
        (
            "reflection_max_output_tokens",
            str(request.max_output_tokens),
        ),
        ("reflection_prompt_sha256", request.prompt_sha256),
        ("reflection_request_sha256", request.request_sha256),
        (
            "reflection_temperature",
            (
                "none"
                if request.temperature is None
                else float(request.temperature).hex()
            ),
        ),
        ("terminal_validation_receipt_sha256", terminal.receipt_sha256),
    ]
    if curation_receipt.failure_type is not None:
        expected_metadata.append(
            ("curation_failure_type", curation_receipt.failure_type)
        )
    if engine_call_receipt.telemetry_sha256 is not None:
        expected_metadata.append(
            (
                "reflection_telemetry_sha256",
                engine_call_receipt.telemetry_sha256,
            )
        )
    expected_metadata_tuple = tuple(sorted(expected_metadata))
    if (
        feedback.generation != 3
        or feedback.policy_id != curation_spec.policy_id
        or feedback.policy_version != curation_spec.policy_version
        or feedback.reservation_hash != expected_reservation.reservation_hash
        or feedback.generation_receipt_hash
        != state.generation_receipts[2].receipt_hash
        or feedback.reserved_logical_llm_calls != 1
        or feedback.used_logical_llm_calls != 1
        or feedback.logical_llm_calls_before != 5
        or feedback.logical_llm_calls_after != 6
        or feedback.result_metadata != expected_metadata_tuple
    ):
        _fail(
            "post-G3 feedback is not bound to expected policy, reservation, "
            "terminal gate, and engine call"
        )
    return G3CausalScreenResultValidationReceipt(
        optimizer_result_sha256=result.result_hash,
        terminal_state_receipt_sha256=terminal.receipt_sha256,
        curation_feedback_receipt_sha256=feedback.receipt_hash,
        curation_authority_sha256=expected_authority.authority_sha256,
        curation_receipt_sha256=curation_receipt.receipt_sha256,
        reflection_call_receipt_sha256=engine_call_receipt.receipt_sha256,
        curation_status=curation_receipt.curation_status,
        curation_publication_outcome=curation_receipt.publication_outcome,
    )


__all__ = [
    "G3CausalScreenResultValidationReceipt",
    "G3MechanismDecision",
    "G3TerminalStateValidationReceipt",
    "G3TerminalValidationError",
    "validate_g3_causal_screen_result",
    "validate_g3_terminal_state",
]
