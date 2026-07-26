"""Generic, authority-bound post-G3 reflection and atomic revision.

The G3 causal screen seals optimization endpoints before this policy runs.
This module then performs one optional provider call whose failure cannot erase
those endpoints.  Every accepted result is joined to an engine-issued
request/provider/publication receipt; interceptor-authored metadata is never
treated as evidence on its own.

Benchmark semantics enter only through an injected ``ReflectionInsightContract``
and a declarative source-slot scope.  This keeps the workflow reusable across
Airfoil, compiler flows, scheduling, or any future finite-action benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationOutcome,
    ReflectionCallExecutionError,
    ReflectionCallReceipt,
    ReflectionCallStatus,
    ReflectionPublicationResult,
)
from agent_evolve.application.budgeted_optimizer import GenerationReceipt
from agent_evolve.application.g3_causal_screen import G3CausalScreenPlanner
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
    InsightOrigin,
)
from agent_evolve.domain.ids import OperatorInvocationId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.agentic_generator import ReflectionInsightContract

if TYPE_CHECKING:
    from agent_evolve.application.g3_causal_validation import (
        G3TerminalStateValidationReceipt,
    )


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SOURCE_SCOPE_DOMAIN = b"agent-evolve:g3-curation-source-scope:v1\x00"
_SPEC_DOMAIN = b"agent-evolve:g3-postseal-curation-spec:v1\x00"
_AUTHORITY_DOMAIN = b"agent-evolve:g3-postseal-curation-authority:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:g3-postseal-curation-receipt:v1\x00"
G3_POSTSEAL_CURATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:g3-postseal-atomic-revision:v1:terminal-gate;"
    b"declarative-source-scope;engine-request-provider-publication-receipt;"
    b"one-zero-prior-quarantined-revision;isolated-provider-failure"
).hexdigest()


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


def _ref_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


@dataclass(frozen=True, slots=True)
class G3CurationSourceScope:
    """Declarative ordered selection of sealed G1--G3 outcome slots.

    Selecting direct adaptive evidence only is useful when an exact finite
    action contract cannot honestly attribute a composite union.  Selecting all
    ten slots is appropriate for family-level reflection.  Either choice is
    explicit, versioned, and included in the curation authority.
    """

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    slot_ids: tuple[str, ...]
    scope_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("source scope policy_id must use the token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("source scope policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "source scope policy_definition_sha256",
        )
        if (
            type(self.slot_ids) is not tuple
            or not self.slot_ids
            or any(
                type(value) is not str
                or not value
                or value != value.strip()
                for value in self.slot_ids
            )
        ):
            raise TypeError("source scope slot_ids must be canonical strings")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("source scope slot_ids cannot repeat")
        object.__setattr__(
            self,
            "scope_sha256",
            _hash(_SOURCE_SCOPE_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "slot_ids": list(self.slot_ids),
        }

    def select(
        self,
        receipts: tuple[GenerationReceipt, ...],
    ) -> tuple[InvocationOutcome, ...]:
        """Resolve exact outcome objects from already-sealed receipts."""

        if type(receipts) is not tuple or len(receipts) != 3:
            raise ValueError("G3 curation requires exactly three generation receipts")
        outcomes_by_slot: dict[str, InvocationOutcome] = {}
        for receipt in receipts:
            if type(receipt) is not GenerationReceipt:
                raise TypeError("source receipts must be exact GenerationReceipt values")
            for slot_result in receipt.slot_results:
                slot_id = slot_result.slot.slot_id
                if slot_id in outcomes_by_slot:
                    raise ValueError("generation receipts repeat a source slot ID")
                outcomes_by_slot[slot_id] = slot_result.outcome
        missing = tuple(
            slot_id for slot_id in self.slot_ids if slot_id not in outcomes_by_slot
        )
        if missing:
            raise ValueError("curation source scope names an absent G1--G3 slot")
        return tuple(outcomes_by_slot[slot_id] for slot_id in self.slot_ids)


@dataclass(frozen=True, slots=True)
class G3PostsealCurationSpec:
    """Benchmark-neutral behavior plus injected reflection vocabulary/scope."""

    insight_contract: ReflectionInsightContract
    source_scope: G3CurationSourceScope
    policy_id: str = "g3_postseal_atomic_revision"
    policy_version: int = 1
    policy_definition_sha256: str = G3_POSTSEAL_CURATION_DEFINITION_SHA256
    label: str = "g3_postseal_curation"
    spec_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.insight_contract) is not ReflectionInsightContract:
            raise TypeError("curation insight_contract must be exact")
        ReflectionInsightContract.__post_init__(self.insight_contract)
        if type(self.source_scope) is not G3CurationSourceScope:
            raise TypeError("curation source_scope must be exact")
        G3CurationSourceScope.__post_init__(self.source_scope)
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("curation policy_id must use the token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("curation policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "curation policy_definition_sha256",
        )
        if (
            type(self.label) is not str
            or not self.label
            or self.label != self.label.strip()
        ):
            raise ValueError("curation label must be canonical non-empty text")
        object.__setattr__(self, "spec_sha256", _hash(_SPEC_DOMAIN, self.to_record()))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "label": self.label,
            "insight_contract_sha256": self.insight_contract.identity_sha256,
            "source_scope_sha256": self.source_scope.scope_sha256,
        }


def _adaptive_predecessor(
    planner: G3CausalScreenPlanner,
    memory: InsightMemoryBank,
) -> tuple[InsightRef, InsightMemoryEntry]:
    if type(planner) is not G3CausalScreenPlanner:
        raise TypeError("curation predecessor planner must be exact")
    if type(memory) is not InsightMemoryBank:
        raise TypeError("curation predecessor memory must be exact")
    assignments = planner.g2_assignments
    if len(assignments) != 2:
        raise ValueError("curation requires the sealed adaptive assignment")
    selected = assignments[0].selection_decision.selected
    if len(selected) != 1:
        raise ValueError("adaptive assignment must select exactly one card")
    reference = selected[0]
    return reference, memory.entries_for((reference,))[0]


def build_g3_postseal_curation_reservation(
    *,
    spec: G3PostsealCurationSpec,
    planner: G3CausalScreenPlanner,
    memory: InsightMemoryBank,
    generation: int,
) -> GenerationFeedbackReservation:
    """Rebuild the expected reservation independently for final validation."""

    if type(spec) is not G3PostsealCurationSpec:
        raise TypeError("curation reservation spec must be exact")
    G3PostsealCurationSpec.__post_init__(spec)
    if type(generation) is not int or not 1 <= generation <= 3:
        raise ValueError("curation reservation generation must lie in [1,3]")
    if generation < 3:
        metadata = tuple(
            sorted(
                (
                    ("curation_spec_sha256", spec.spec_sha256),
                    ("stage", "sealed_no_op"),
                )
            )
        )
    else:
        terminal_authority = planner.terminal_validation_authority
        if terminal_authority is None:
            raise ValueError("G3 terminal validation authority is not sealed")
        predecessor, entry = _adaptive_predecessor(planner, memory)
        metadata = tuple(
            sorted(
                (
                    ("curation_spec_sha256", spec.spec_sha256),
                    (
                        "reflection_contract_sha256",
                        spec.insight_contract.identity_sha256,
                    ),
                    (
                        "revision_predecessor",
                        f"{predecessor.insight_id.value}@{predecessor.version}",
                    ),
                    (
                        "revision_predecessor_content_sha256",
                        entry.draft.content_sha256,
                    ),
                    ("source_scope_sha256", spec.source_scope.scope_sha256),
                    ("stage", "post_g3"),
                    (
                        "terminal_validation_authority_sha256",
                        terminal_authority.authority_sha256,
                    ),
                )
            )
        )
    return GenerationFeedbackReservation(
        policy_id=spec.policy_id,
        policy_version=spec.policy_version,
        logical_llm_calls=1 if generation == 3 else 0,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class G3PostsealCurationAuthority:
    """Pre-provider commitment joining the terminal gate to one request."""

    spec_sha256: str
    reservation_hash: str
    terminal_validation_receipt_sha256: str
    terminal_validation_authority_sha256: str
    generation_receipt_sha256s: tuple[str, str, str]
    source_scope_sha256: str
    source_slot_ids: tuple[str, ...]
    source_operator_invocation_ids: tuple[OperatorInvocationId, ...]
    revision_predecessor: InsightRef
    revision_predecessor_content_sha256: str
    insight_contract_sha256: str
    reflection_label: str
    authority_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "spec_sha256",
            "reservation_hash",
            "terminal_validation_receipt_sha256",
            "terminal_validation_authority_sha256",
            "source_scope_sha256",
            "revision_predecessor_content_sha256",
            "insight_contract_sha256",
        ):
            require_sha256(getattr(self, name), f"curation authority {name}")
        if (
            type(self.generation_receipt_sha256s) is not tuple
            or len(self.generation_receipt_sha256s) != 3
        ):
            raise ValueError("curation authority requires G1--G3 receipt hashes")
        for value in self.generation_receipt_sha256s:
            require_sha256(value, "curation source generation receipt")
        if (
            type(self.source_slot_ids) is not tuple
            or not self.source_slot_ids
            or len(set(self.source_slot_ids)) != len(self.source_slot_ids)
        ):
            raise ValueError("curation authority requires unique source slots")
        if type(self.source_operator_invocation_ids) is not tuple or any(
            type(value) is not OperatorInvocationId
            for value in self.source_operator_invocation_ids
        ):
            raise TypeError("curation source operator IDs must be exact")
        for value in self.source_operator_invocation_ids:
            OperatorInvocationId.__post_init__(value)
        if len(self.source_slot_ids) != len(self.source_operator_invocation_ids):
            raise ValueError("curation source slots/operator IDs differ")
        if len(set(self.source_operator_invocation_ids)) != len(
            self.source_operator_invocation_ids
        ):
            raise ValueError("curation source operator IDs cannot repeat")
        if type(self.revision_predecessor) is not InsightRef:
            raise TypeError("curation revision predecessor must be exact")
        InsightRef.__post_init__(self.revision_predecessor)
        if (
            type(self.reflection_label) is not str
            or not self.reflection_label
            or self.reflection_label != self.reflection_label.strip()
        ):
            raise ValueError("curation reflection label must be canonical")
        expected = _hash(_AUTHORITY_DOMAIN, self.to_record())
        if self.authority_sha256:
            require_sha256(self.authority_sha256, "curation authority_sha256")
            if self.authority_sha256 != expected:
                raise ValueError("curation authority hash does not authenticate data")
        else:
            object.__setattr__(self, "authority_sha256", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "spec_sha256": self.spec_sha256,
            "reservation_hash": self.reservation_hash,
            "terminal_validation_receipt_sha256": (
                self.terminal_validation_receipt_sha256
            ),
            "terminal_validation_authority_sha256": (
                self.terminal_validation_authority_sha256
            ),
            "generation_receipt_sha256s": list(
                self.generation_receipt_sha256s
            ),
            "source_scope_sha256": self.source_scope_sha256,
            "source_slot_ids": list(self.source_slot_ids),
            "source_operator_invocation_ids": [
                value.value for value in self.source_operator_invocation_ids
            ],
            "revision_predecessor": _ref_record(self.revision_predecessor),
            "revision_predecessor_content_sha256": (
                self.revision_predecessor_content_sha256
            ),
            "insight_contract_sha256": self.insight_contract_sha256,
            "reflection_label": self.reflection_label,
            "requested_cardinality": {"min": 0, "max": 1},
        }


@dataclass(frozen=True, slots=True)
class G3PostsealCurationReceipt:
    """Authenticated success/failure result for the isolated sixth call."""

    authority: G3PostsealCurationAuthority
    call_receipt: ReflectionCallReceipt
    curation_status: str
    failure_type: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.authority) is not G3PostsealCurationAuthority:
            raise TypeError("curation receipt authority must be exact")
        G3PostsealCurationAuthority.__post_init__(self.authority)
        if type(self.call_receipt) is not ReflectionCallReceipt:
            raise TypeError("curation call_receipt must be exact")
        ReflectionCallReceipt.__post_init__(self.call_receipt)
        authority = self.authority
        request = self.call_receipt.request
        if (
            request.label != authority.reflection_label
            or request.operation != "extract_insights"
            or request.min_insights != 0
            or request.max_insights != 1
            or request.insight_contract_sha256
            != authority.insight_contract_sha256
            or request.revision_predecessors
            != (authority.revision_predecessor,)
            or request.revision_predecessor_content_sha256s
            != (authority.revision_predecessor_content_sha256,)
            or request.source_receipt_sha256s
            != authority.generation_receipt_sha256s
            or request.source_operator_invocation_ids
            != authority.source_operator_invocation_ids
            or len(request.source_outcome_sha256s)
            != len(authority.source_slot_ids)
        ):
            raise ValueError("engine reflection request differs from curation authority")
        publications = self.call_receipt.publications
        if self.curation_status == "sealed_complete":
            if (
                self.failure_type is not None
                or self.call_receipt.status is not ReflectionCallStatus.COMPLETED
                or len(publications) > 1
            ):
                raise ValueError("completed curation has invalid provider/publication data")
            if publications:
                publication = publications[0]
                predecessor = authority.revision_predecessor
                if (
                    publication.revision_predecessor != predecessor
                    or publication.reference.insight_id != predecessor.insight_id
                    or publication.reference.version != predecessor.version + 1
                    or publication.lifecycle_state
                    is not InsightLifecycleState.QUARANTINED
                    or publication.origin is not InsightOrigin.REFLECTION
                    or publication.initial_score != 0.0
                ):
                    raise ValueError("curation publication is not one fresh revision")
        elif self.curation_status == "incomplete":
            if (
                type(self.failure_type) is not str
                or not self.failure_type
                or self.call_receipt.status is not ReflectionCallStatus.FAILED
                or self.call_receipt.failure_type != self.failure_type
                or publications
            ):
                raise ValueError("incomplete curation lacks exact failure evidence")
        else:
            raise ValueError("curation_status must be sealed_complete or incomplete")
        expected = _hash(_RECEIPT_DOMAIN, self.to_record())
        if self.receipt_sha256:
            require_sha256(self.receipt_sha256, "curation receipt_sha256")
            if self.receipt_sha256 != expected:
                raise ValueError("curation receipt hash does not authenticate data")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "authority_sha256": self.authority.authority_sha256,
            "reflection_call_receipt_sha256": self.call_receipt.receipt_sha256,
            "reflection_call_id": self.call_receipt.call_id.value,
            "reflection_request_sha256": self.call_receipt.request.request_sha256,
            "curation_status": self.curation_status,
            "publication_outcome": self.publication_outcome,
            "failure_type": self.failure_type,
            "publication_sha256s": [
                value.publication_sha256
                for value in self.call_receipt.publications
            ],
        }

    @property
    def publication_outcome(self) -> str:
        if self.curation_status == "incomplete":
            return "failed"
        return (
            "completed_revision"
            if self.call_receipt.publications
            else "completed_abstention"
        )


class G3PostsealCurationInterceptor:
    """One reusable feedback policy for the exact generic G3 screen."""

    def __init__(
        self,
        *,
        engine: AgenticEvolutionEngine,
        planner: G3CausalScreenPlanner,
        memory: InsightMemoryBank,
        spec: G3PostsealCurationSpec,
    ) -> None:
        if type(engine) is not AgenticEvolutionEngine:
            raise TypeError("curation engine must be exact")
        if type(planner) is not G3CausalScreenPlanner:
            raise TypeError("curation planner must be exact")
        if type(memory) is not InsightMemoryBank:
            raise TypeError("curation memory must be exact")
        if type(spec) is not G3PostsealCurationSpec:
            raise TypeError("curation spec must be exact")
        G3PostsealCurationSpec.__post_init__(spec)
        if planner.engine is not engine or planner.memory is not memory:
            raise ValueError("curation collaborators differ from planner runtime")
        self.engine = engine
        self.planner = planner
        self.memory = memory
        self.spec = spec
        self.policy_id = spec.policy_id
        self.policy_version = spec.policy_version
        self.invoked_generations: list[int] = []
        self.curated_entries: tuple[InsightMemoryEntry, ...] = ()
        self.terminal_validation_receipt: G3TerminalStateValidationReceipt | None = None
        self.curation_authority: G3PostsealCurationAuthority | None = None
        self.curation_receipt: G3PostsealCurationReceipt | None = None
        self.curation_failure_type: str | None = None
        self._reservations: dict[int, GenerationFeedbackReservation] = {}

    def _predecessor(self) -> tuple[InsightRef, InsightMemoryEntry]:
        return _adaptive_predecessor(self.planner, self.memory)

    def _reservation_metadata(self, *, generation: int) -> tuple[tuple[str, str], ...]:
        if generation < 3:
            return tuple(
                sorted(
                    (
                        ("curation_spec_sha256", self.spec.spec_sha256),
                        ("stage", "sealed_no_op"),
                    )
                )
            )
        if generation != 3:
            raise ValueError("G3 curation cannot reserve a foreign generation")
        terminal_authority = self.planner.terminal_validation_authority
        if terminal_authority is None:
            raise ValueError("G3 terminal validation authority is not sealed")
        predecessor, entry = self._predecessor()
        return tuple(
            sorted(
                (
                    ("curation_spec_sha256", self.spec.spec_sha256),
                    (
                        "reflection_contract_sha256",
                        self.spec.insight_contract.identity_sha256,
                    ),
                    (
                        "revision_predecessor",
                        f"{predecessor.insight_id.value}@{predecessor.version}",
                    ),
                    (
                        "revision_predecessor_content_sha256",
                        entry.draft.content_sha256,
                    ),
                    ("source_scope_sha256", self.spec.source_scope.scope_sha256),
                    ("stage", "post_g3"),
                    (
                        "terminal_validation_authority_sha256",
                        terminal_authority.authority_sha256,
                    ),
                )
            )
        )

    def expected_reservation(self, *, generation: int) -> GenerationFeedbackReservation:
        reservation = build_g3_postseal_curation_reservation(
            spec=self.spec,
            planner=self.planner,
            memory=self.memory,
            generation=generation,
        )
        if reservation.metadata != self._reservation_metadata(generation=generation):
            raise AssertionError("curation reservation implementations diverged")
        return reservation

    def reserve(self, *, state, plan) -> GenerationFeedbackReservation:
        if state.generation + 1 != plan.generation:
            raise ValueError("feedback reservation observed the wrong generation")
        if plan.generation in self._reservations:
            raise RuntimeError("curation generation was reserved twice")
        reservation = self.expected_reservation(generation=plan.generation)
        self._reservations[plan.generation] = reservation
        return reservation

    async def after_generation(
        self,
        context: GenerationFeedbackContext,
    ) -> GenerationFeedbackResult:
        generation = context.plan.generation
        self.invoked_generations.append(generation)
        expected_reservation = self._reservations.get(generation)
        if context.reservation != expected_reservation:
            raise ValueError("curation context differs from its frozen reservation")
        if generation < 3:
            if context.reservation.logical_llm_calls != 0:
                raise ValueError("early G3 feedback reservation must be zero")
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=tuple(
                    sorted(
                        (
                            ("curation_spec_sha256", self.spec.spec_sha256),
                            ("curation_status", "not_due"),
                        )
                    )
                ),
            )
        if generation != 3:
            raise ValueError("G3 curation received a foreign generation")
        from agent_evolve.application.g3_causal_validation import (
            validate_g3_terminal_state,
        )

        self.terminal_validation_receipt = validate_g3_terminal_state(
            state=context.state,
            planner=self.planner,
            evaluation_cache_snapshot=(
                await self.engine.evaluation_cache_snapshot()
            ),
        )
        receipts = context.state.generation_receipts
        if (
            type(receipts) is not tuple
            or len(receipts) != 3
            or receipts[-1] != context.generation_receipt
        ):
            raise ValueError("curation requires sealed G1 through G3 receipts")
        outcomes = self.spec.source_scope.select(receipts)
        predecessor, predecessor_entry = self._predecessor()
        terminal_authority = self.planner.terminal_validation_authority
        if terminal_authority is None:
            raise ValueError("terminal authority disappeared before curation")
        generation_receipt_sha256s = tuple(
            receipt.receipt_hash for receipt in receipts
        )
        if len(generation_receipt_sha256s) != 3:  # narrows the tuple type.
            raise AssertionError("G3 receipt cardinality changed")
        authority = G3PostsealCurationAuthority(
            spec_sha256=self.spec.spec_sha256,
            reservation_hash=context.reservation.reservation_hash,
            terminal_validation_receipt_sha256=(
                self.terminal_validation_receipt.receipt_sha256
            ),
            terminal_validation_authority_sha256=(
                terminal_authority.authority_sha256
            ),
            generation_receipt_sha256s=generation_receipt_sha256s,
            source_scope_sha256=self.spec.source_scope.scope_sha256,
            source_slot_ids=self.spec.source_scope.slot_ids,
            source_operator_invocation_ids=tuple(
                outcome.prepared.operator_invocation_id for outcome in outcomes
            ),
            revision_predecessor=predecessor,
            revision_predecessor_content_sha256=(
                predecessor_entry.draft.content_sha256
            ),
            insight_contract_sha256=(
                self.spec.insight_contract.identity_sha256
            ),
            reflection_label=self.spec.label,
        )
        # Publish the authority before crossing the provider boundary.  A typed
        # provider/cardinality failure can therefore be isolated and audited.
        self.curation_authority = authority
        try:
            publication_result = await self.engine.reflect_with_receipt(
                outcomes,
                label=self.spec.label,
                max_insights=1,
                min_insights=0,
                insight_contract=self.spec.insight_contract,
                revision_predecessors=(predecessor,),
                source_receipt_sha256s=generation_receipt_sha256s,
            )
        except ReflectionCallExecutionError as exc:
            self.curation_failure_type = exc.failure_type
            receipt = G3PostsealCurationReceipt(
                authority=authority,
                call_receipt=exc.receipt,
                curation_status="incomplete",
                failure_type=exc.failure_type,
            )
            self.curation_receipt = receipt
            return self._feedback_result(receipt)
        if type(publication_result) is not ReflectionPublicationResult:
            raise TypeError("receipt-bearing reflection returned a foreign result")
        ReflectionPublicationResult.__post_init__(publication_result)
        self.curated_entries = publication_result.entries
        receipt = G3PostsealCurationReceipt(
            authority=authority,
            call_receipt=publication_result.receipt,
            curation_status="sealed_complete",
            failure_type=None,
        )
        self.curation_receipt = receipt
        return self._feedback_result(receipt)

    def _feedback_result(
        self,
        receipt: G3PostsealCurationReceipt,
    ) -> GenerationFeedbackResult:
        metadata = [
            ("curated_entry_count", str(len(receipt.call_receipt.publications))),
            ("curation_authority_sha256", receipt.authority.authority_sha256),
            ("curation_receipt_sha256", receipt.receipt_sha256),
            ("curation_spec_sha256", self.spec.spec_sha256),
            ("curation_status", receipt.curation_status),
            ("curation_publication_outcome", receipt.publication_outcome),
            (
                "reflection_call_receipt_sha256",
                receipt.call_receipt.receipt_sha256,
            ),
            (
                "reflection_call_id",
                receipt.call_receipt.call_id.value,
            ),
            (
                "reflection_max_output_tokens",
                str(receipt.call_receipt.request.max_output_tokens),
            ),
            (
                "reflection_prompt_sha256",
                receipt.call_receipt.request.prompt_sha256,
            ),
            (
                "reflection_request_sha256",
                receipt.call_receipt.request.request_sha256,
            ),
            (
                "reflection_temperature",
                (
                    "none"
                    if receipt.call_receipt.request.temperature is None
                    else float(
                        receipt.call_receipt.request.temperature
                    ).hex()
                ),
            ),
            (
                "terminal_validation_receipt_sha256",
                receipt.authority.terminal_validation_receipt_sha256,
            ),
        ]
        if receipt.failure_type is not None:
            metadata.append(("curation_failure_type", receipt.failure_type))
        if receipt.call_receipt.telemetry_sha256 is not None:
            metadata.append(
                (
                    "reflection_telemetry_sha256",
                    receipt.call_receipt.telemetry_sha256,
                )
            )
        return GenerationFeedbackResult(
            logical_llm_calls_used=1,
            metadata=tuple(sorted(metadata)),
        )


class G3PostsealCurationFactory:
    """Deferred public-composition factory for the exact runtime identities."""

    def __init__(
        self,
        *,
        spec: G3PostsealCurationSpec,
    ) -> None:
        if type(spec) is not G3PostsealCurationSpec:
            raise TypeError("curation factory spec must be exact")
        G3PostsealCurationSpec.__post_init__(spec)
        self.spec = spec
        self.interceptor: G3PostsealCurationInterceptor | None = None
        self.runtime_identities: tuple[object, object, object, object] | None = None

    def build(self, *, benchmark, engine, id_factory, memory, planner):
        if self.interceptor is not None:
            raise RuntimeError("curation factory may be invoked only once")
        if type(planner) is not G3CausalScreenPlanner:
            raise TypeError("curation factory requires the exact G3 runtime planner")
        if (
            planner.benchmark is not benchmark
            or planner.engine is not engine
            or planner.ids is not id_factory
            or planner.memory is not memory
        ):
            raise ValueError("curation factory received foreign runtime identities")
        self.runtime_identities = (benchmark, engine, id_factory, memory)
        self.interceptor = G3PostsealCurationInterceptor(
            engine=engine,
            planner=planner,
            memory=memory,
            spec=self.spec,
        )
        return self.interceptor


__all__ = [
    "G3CurationSourceScope",
    "G3PostsealCurationAuthority",
    "G3PostsealCurationFactory",
    "G3PostsealCurationInterceptor",
    "G3PostsealCurationReceipt",
    "G3PostsealCurationSpec",
    "G3_POSTSEAL_CURATION_DEFINITION_SHA256",
    "build_g3_postseal_curation_reservation",
]
