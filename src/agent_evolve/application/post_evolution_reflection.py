"""Benchmark-neutral, receipt-bound reflection after a terminal generation.

The optimizer seals a complete generation before invoking a feedback
interceptor.  This module uses that boundary to run at most one optional
reflection call after a configured terminal generation.  Earlier generations
reserve no calls.  The terminal call may revise one planner-selected memory
entry and can inspect any explicitly named ``(generation, slot_id)`` outcomes.

Nothing in this module knows a benchmark's metrics, actions, planner type, or
generation count.  Those semantics enter through an injected
``ReflectionInsightContract``, a declarative source scope, and a predecessor
resolver.  Provider failures are represented by an authenticated incomplete
receipt and do not erase the already-sealed optimization outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationOutcome,
    ReflectionCallExecutionError,
    ReflectionCallReceipt,
    ReflectionCallStatus,
    ReflectionPublicationResult,
)
from agent_evolve.application.budgeted_optimizer import (
    GenerationPlan,
    GenerationReceipt,
    OptimizerState,
    validate_generation_receipt_integrity,
)
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


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SOURCE_SCOPE_DOMAIN = b"agent-evolve:post-evolution-reflection-source-scope:v1\x00"
_SPEC_DOMAIN = b"agent-evolve:post-evolution-reflection-spec:v1\x00"
_AUTHORITY_DOMAIN = b"agent-evolve:post-evolution-reflection-authority:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:post-evolution-reflection-receipt:v1\x00"

POST_EVOLUTION_REFLECTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:post-evolution-reflection:v1;terminal-only-one-call;"
    b"generation-slot-source-scope;planner-resolved-predecessor;"
    b"injected-reflection-contract;engine-issued-request-publication-receipt;"
    b"zero-prior-quarantined-revision;isolated-provider-failure"
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


@dataclass(frozen=True, slots=True, order=True)
class PostEvolutionReflectionSource:
    """One exact outcome coordinate in the sealed generation ledger."""

    generation: int
    slot_id: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("reflection source generation must be positive")
        if (
            type(self.slot_id) is not str
            or not self.slot_id
            or self.slot_id != self.slot_id.strip()
        ):
            raise ValueError("reflection source slot_id must be canonical")

    def to_record(self) -> dict[str, object]:
        return {"generation": self.generation, "slot_id": self.slot_id}


def _validated_receipts(
    receipts: tuple[GenerationReceipt, ...],
) -> tuple[GenerationReceipt, ...]:
    if type(receipts) is not tuple or not receipts:
        raise ValueError("reflection sources require sealed generation receipts")
    if any(type(receipt) is not GenerationReceipt for receipt in receipts):
        raise TypeError("source receipts must be exact GenerationReceipt values")
    for receipt in receipts:
        validate_generation_receipt_integrity(receipt)
    generations = tuple(receipt.generation for receipt in receipts)
    if generations != tuple(range(1, len(receipts) + 1)):
        raise ValueError("source receipts must be contiguous and ordered")
    return receipts


@dataclass(frozen=True, slots=True)
class PostEvolutionReflectionSourceScope:
    """Ordered, versioned selection of outcomes from sealed receipts.

    Slot identifiers only need to be unique within a generation.  Pairing each
    identifier with its generation therefore permits stable scopes even when a
    generic planner intentionally reuses role names across generations.
    """

    sources: tuple[PostEvolutionReflectionSource, ...]
    policy_id: str = "explicit_generation_slot_scope"
    policy_version: int = 1
    policy_definition_sha256: str = POST_EVOLUTION_REFLECTION_DEFINITION_SHA256
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
        if type(self.sources) is not tuple or not self.sources:
            raise ValueError("source scope requires at least one source")
        if any(
            type(source) is not PostEvolutionReflectionSource for source in self.sources
        ):
            raise TypeError("source scope must contain exact source coordinates")
        for source in self.sources:
            PostEvolutionReflectionSource.__post_init__(source)
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("source scope cannot repeat a generation/slot pair")
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
            "sources": [source.to_record() for source in self.sources],
        }

    def select(
        self,
        receipts: tuple[GenerationReceipt, ...],
    ) -> tuple[InvocationOutcome, ...]:
        """Return exact outcomes in the declared source order."""

        sealed = _validated_receipts(receipts)
        outcomes: list[InvocationOutcome] = []
        for source in self.sources:
            if source.generation > len(sealed):
                raise ValueError("source scope names an unsealed generation")
            receipt = sealed[source.generation - 1]
            matches = tuple(
                slot_result.outcome
                for slot_result in receipt.slot_results
                if slot_result.slot.slot_id == source.slot_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "source scope must resolve each generation/slot pair exactly once"
                )
            outcome = matches[0]
            if type(outcome) is not InvocationOutcome:
                raise TypeError("source scope resolved a foreign invocation outcome")
            outcomes.append(outcome)
        return tuple(outcomes)


@runtime_checkable
class PostEvolutionPredecessorResolver(Protocol):
    """Resolve the memory entry selected by a benchmark-owned planner.

    Composition invokes the resolver only after ``planner.plan`` has completed
    for the terminal generation.  A resolver may therefore read a planner's
    frozen memory-assignment receipt without coupling this generic policy to
    the planner's concrete type.
    """

    def __call__(self, planner: object) -> InsightRef: ...


@dataclass(frozen=True, slots=True)
class PostEvolutionReflectionSpec:
    """Complete benchmark-injected policy for one terminal reflection."""

    terminal_generation: int
    source_scope: PostEvolutionReflectionSourceScope
    insight_contract: ReflectionInsightContract
    policy_id: str = "post_evolution_atomic_revision"
    policy_version: int = 1
    policy_definition_sha256: str = POST_EVOLUTION_REFLECTION_DEFINITION_SHA256
    label: str = "post_evolution_reflection"
    spec_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.terminal_generation) is not int or self.terminal_generation <= 0:
            raise ValueError("terminal_generation must be positive")
        if type(self.source_scope) is not PostEvolutionReflectionSourceScope:
            raise TypeError("reflection source_scope must be exact")
        PostEvolutionReflectionSourceScope.__post_init__(self.source_scope)
        if any(
            source.generation > self.terminal_generation
            for source in self.source_scope.sources
        ):
            raise ValueError("reflection source cannot follow the terminal generation")
        if type(self.insight_contract) is not ReflectionInsightContract:
            raise TypeError("reflection insight_contract must be exact")
        ReflectionInsightContract.__post_init__(self.insight_contract)
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("reflection policy_id must use the token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("reflection policy_version must be positive")
        require_sha256(
            self.policy_definition_sha256,
            "reflection policy_definition_sha256",
        )
        if (
            type(self.label) is not str
            or not self.label
            or self.label != self.label.strip()
        ):
            raise ValueError("reflection label must be canonical non-empty text")
        object.__setattr__(self, "spec_sha256", _hash(_SPEC_DOMAIN, self.to_record()))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "terminal_generation": self.terminal_generation,
            "source_scope_sha256": self.source_scope.scope_sha256,
            "insight_contract_sha256": self.insight_contract.identity_sha256,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "label": self.label,
            "requested_cardinality": {"min": 0, "max": 1},
        }


@dataclass(frozen=True, slots=True)
class PostEvolutionReflectionAuthority:
    """Pre-provider commitment to exact sealed evidence and one revision."""

    spec_sha256: str
    reservation_hash: str
    terminal_generation: int
    generation_receipt_sha256s: tuple[str, ...]
    source_scope_sha256: str
    sources: tuple[PostEvolutionReflectionSource, ...]
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
            "source_scope_sha256",
            "revision_predecessor_content_sha256",
            "insight_contract_sha256",
        ):
            require_sha256(getattr(self, name), f"reflection authority {name}")
        if type(self.terminal_generation) is not int or self.terminal_generation <= 0:
            raise ValueError("authority terminal_generation must be positive")
        if (
            type(self.generation_receipt_sha256s) is not tuple
            or len(self.generation_receipt_sha256s) != self.terminal_generation
        ):
            raise ValueError("authority requires every terminal generation receipt")
        for value in self.generation_receipt_sha256s:
            require_sha256(value, "reflection source generation receipt")
        if type(self.sources) is not tuple or not self.sources:
            raise ValueError("reflection authority requires source coordinates")
        if any(
            type(value) is not PostEvolutionReflectionSource for value in self.sources
        ):
            raise TypeError("reflection authority sources must be exact")
        for value in self.sources:
            PostEvolutionReflectionSource.__post_init__(value)
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("reflection authority sources cannot repeat")
        if any(value.generation > self.terminal_generation for value in self.sources):
            raise ValueError("reflection authority source follows terminal generation")
        if type(self.source_operator_invocation_ids) is not tuple or any(
            type(value) is not OperatorInvocationId
            for value in self.source_operator_invocation_ids
        ):
            raise TypeError("reflection source operator IDs must be exact")
        for value in self.source_operator_invocation_ids:
            OperatorInvocationId.__post_init__(value)
        if len(self.sources) != len(self.source_operator_invocation_ids):
            raise ValueError("reflection source coordinates/operator IDs differ")
        if len(set(self.source_operator_invocation_ids)) != len(
            self.source_operator_invocation_ids
        ):
            raise ValueError("reflection source operator IDs cannot repeat")
        if type(self.revision_predecessor) is not InsightRef:
            raise TypeError("reflection revision_predecessor must be exact")
        InsightRef.__post_init__(self.revision_predecessor)
        if (
            type(self.reflection_label) is not str
            or not self.reflection_label
            or self.reflection_label != self.reflection_label.strip()
        ):
            raise ValueError("authority reflection_label must be canonical")
        expected = _hash(_AUTHORITY_DOMAIN, self.to_record())
        if self.authority_sha256:
            require_sha256(self.authority_sha256, "reflection authority_sha256")
            if self.authority_sha256 != expected:
                raise ValueError("reflection authority hash does not authenticate data")
        else:
            object.__setattr__(self, "authority_sha256", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "spec_sha256": self.spec_sha256,
            "reservation_hash": self.reservation_hash,
            "terminal_generation": self.terminal_generation,
            "generation_receipt_sha256s": list(self.generation_receipt_sha256s),
            "source_scope_sha256": self.source_scope_sha256,
            "sources": [source.to_record() for source in self.sources],
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
class PostEvolutionReflectionReceipt:
    """Authenticated completion or isolated failure for the terminal call."""

    authority: PostEvolutionReflectionAuthority
    call_receipt: ReflectionCallReceipt
    reflection_status: str
    failure_type: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.authority) is not PostEvolutionReflectionAuthority:
            raise TypeError("reflection receipt authority must be exact")
        PostEvolutionReflectionAuthority.__post_init__(self.authority)
        if type(self.call_receipt) is not ReflectionCallReceipt:
            raise TypeError("reflection call_receipt must be exact")
        ReflectionCallReceipt.__post_init__(self.call_receipt)
        authority = self.authority
        request = self.call_receipt.request
        if (
            request.label != authority.reflection_label
            or request.operation != "extract_insights"
            or request.min_insights != 0
            or request.max_insights != 1
            or request.insight_contract_sha256 != authority.insight_contract_sha256
            or request.revision_predecessors != (authority.revision_predecessor,)
            or request.revision_predecessor_content_sha256s
            != (authority.revision_predecessor_content_sha256,)
            or request.source_receipt_sha256s != authority.generation_receipt_sha256s
            or request.source_operator_invocation_ids
            != authority.source_operator_invocation_ids
            or len(request.source_outcome_sha256s) != len(authority.sources)
        ):
            raise ValueError("engine reflection request differs from its authority")
        publications = self.call_receipt.publications
        if self.reflection_status == "sealed_complete":
            if (
                self.failure_type is not None
                or self.call_receipt.status is not ReflectionCallStatus.COMPLETED
                or len(publications) > 1
            ):
                raise ValueError("completed reflection has invalid call evidence")
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
                    raise ValueError("reflection publication is not one fresh revision")
        elif self.reflection_status == "incomplete":
            if (
                type(self.failure_type) is not str
                or not self.failure_type
                or self.call_receipt.status is not ReflectionCallStatus.FAILED
                or self.call_receipt.failure_type != self.failure_type
                or publications
            ):
                raise ValueError("incomplete reflection lacks exact failure evidence")
        else:
            raise ValueError("reflection_status must be sealed_complete or incomplete")
        expected = _hash(_RECEIPT_DOMAIN, self.to_record())
        if self.receipt_sha256:
            require_sha256(self.receipt_sha256, "reflection receipt_sha256")
            if self.receipt_sha256 != expected:
                raise ValueError("reflection receipt hash does not authenticate data")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    @property
    def publication_outcome(self) -> str:
        if self.reflection_status == "incomplete":
            return "failed"
        return (
            "completed_revision"
            if self.call_receipt.publications
            else "completed_abstention"
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "authority_sha256": self.authority.authority_sha256,
            "reflection_call_receipt_sha256": self.call_receipt.receipt_sha256,
            "reflection_call_id": self.call_receipt.call_id.value,
            "reflection_request_sha256": self.call_receipt.request.request_sha256,
            "reflection_status": self.reflection_status,
            "publication_outcome": self.publication_outcome,
            "failure_type": self.failure_type,
            "publication_sha256s": [
                value.publication_sha256 for value in self.call_receipt.publications
            ],
        }


class PostEvolutionReflectionInterceptor:
    """Reserve zero early calls and exactly one terminal reflection call."""

    def __init__(
        self,
        *,
        engine: AgenticEvolutionEngine,
        planner: object,
        memory: InsightMemoryBank,
        spec: PostEvolutionReflectionSpec,
        predecessor_resolver: PostEvolutionPredecessorResolver,
    ) -> None:
        if type(engine) is not AgenticEvolutionEngine:
            raise TypeError("post-evolution reflection engine must be exact")
        if type(memory) is not InsightMemoryBank:
            raise TypeError("post-evolution reflection memory must be exact")
        if engine.memory is not memory:
            raise ValueError("post-evolution reflection memory differs from engine")
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("post-evolution reflection planner must implement plan")
        if type(spec) is not PostEvolutionReflectionSpec:
            raise TypeError("post-evolution reflection spec must be exact")
        PostEvolutionReflectionSpec.__post_init__(spec)
        if not callable(predecessor_resolver):
            raise TypeError("predecessor_resolver must be callable")
        self.engine = engine
        self.planner = planner
        self.memory = memory
        self.spec = spec
        self.predecessor_resolver = predecessor_resolver
        self.policy_id = spec.policy_id
        self.policy_version = spec.policy_version
        self.invoked_generations: list[int] = []
        self.reflected_entries: tuple[InsightMemoryEntry, ...] = ()
        self.reflection_result: ReflectionPublicationResult | None = None
        self.reflection_authority: PostEvolutionReflectionAuthority | None = None
        self.reflection_receipt: PostEvolutionReflectionReceipt | None = None
        self.reflection_failure_type: str | None = None
        self._reservations: dict[int, GenerationFeedbackReservation] = {}
        self._completed_generations: set[int] = set()
        self._terminal_predecessor: tuple[InsightRef, str] | None = None

    def _resolve_predecessor(self) -> tuple[InsightRef, InsightMemoryEntry]:
        reference = self.predecessor_resolver(self.planner)
        if type(reference) is not InsightRef:
            raise TypeError("predecessor_resolver must return an exact InsightRef")
        InsightRef.__post_init__(reference)
        entry = self.memory.entries_for((reference,))[0]
        if type(entry) is not InsightMemoryEntry:
            raise TypeError("predecessor resolver selected a foreign memory entry")
        InsightMemoryEntry.__post_init__(entry)
        return reference, entry

    def _reservation(
        self,
        *,
        generation: int,
    ) -> GenerationFeedbackReservation:
        terminal = self.spec.terminal_generation
        if type(generation) is not int or not 1 <= generation <= terminal:
            raise ValueError("reflection reservation generation is outside its run")
        if generation < terminal:
            metadata = tuple(
                sorted(
                    (
                        ("reflection_spec_sha256", self.spec.spec_sha256),
                        ("stage", "sealed_no_op"),
                    )
                )
            )
            logical_calls = 0
        else:
            predecessor, entry = self._resolve_predecessor()
            frozen = (predecessor, entry.draft.content_sha256)
            if self._terminal_predecessor is not None:
                raise RuntimeError("terminal reflection predecessor was resolved twice")
            self._terminal_predecessor = frozen
            metadata = tuple(
                sorted(
                    (
                        (
                            "reflection_contract_sha256",
                            self.spec.insight_contract.identity_sha256,
                        ),
                        ("reflection_spec_sha256", self.spec.spec_sha256),
                        (
                            "revision_predecessor",
                            f"{predecessor.insight_id.value}@{predecessor.version}",
                        ),
                        (
                            "revision_predecessor_content_sha256",
                            entry.draft.content_sha256,
                        ),
                        ("source_scope_sha256", self.spec.source_scope.scope_sha256),
                        ("stage", "post_terminal_generation"),
                    )
                )
            )
            logical_calls = 1
        return GenerationFeedbackReservation(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            logical_llm_calls=logical_calls,
            metadata=metadata,
        )

    def reserve(
        self,
        *,
        state: OptimizerState,
        plan: GenerationPlan,
    ) -> GenerationFeedbackReservation:
        if type(state) is not OptimizerState:
            raise TypeError("reflection reservation state must be exact")
        if type(plan) is not GenerationPlan:
            raise TypeError("reflection reservation plan must be exact")
        if state.generation + 1 != plan.generation:
            raise ValueError("reflection reservation observed the wrong generation")
        if plan.generation in self._reservations:
            raise RuntimeError("reflection generation was reserved twice")
        reservation = self._reservation(generation=plan.generation)
        self._reservations[plan.generation] = reservation
        return reservation

    async def after_generation(
        self,
        context: GenerationFeedbackContext,
    ) -> GenerationFeedbackResult:
        if type(context) is not GenerationFeedbackContext:
            raise TypeError("post-evolution reflection context must be exact")
        GenerationFeedbackContext.__post_init__(context)
        generation = context.plan.generation
        if generation in self._completed_generations:
            raise RuntimeError("reflection feedback generation was completed twice")
        expected_reservation = self._reservations.get(generation)
        if context.reservation != expected_reservation:
            raise ValueError("reflection context differs from its frozen reservation")
        self._completed_generations.add(generation)
        self.invoked_generations.append(generation)
        terminal = self.spec.terminal_generation
        if generation < terminal:
            if context.reservation.logical_llm_calls != 0:
                raise ValueError("early reflection reservation must be zero")
            return GenerationFeedbackResult(
                logical_llm_calls_used=0,
                metadata=tuple(
                    sorted(
                        (
                            ("reflection_spec_sha256", self.spec.spec_sha256),
                            ("reflection_status", "not_due"),
                        )
                    )
                ),
            )
        if generation != terminal:
            raise ValueError("reflection interceptor received a foreign generation")
        if context.reservation.logical_llm_calls != 1:
            raise ValueError("terminal reflection must reserve exactly one call")
        receipts = context.state.generation_receipts
        if (
            type(receipts) is not tuple
            or len(receipts) != terminal
            or receipts[-1] != context.generation_receipt
        ):
            raise ValueError("terminal reflection requires every sealed receipt")
        _validated_receipts(receipts)
        outcomes = self.spec.source_scope.select(receipts)
        frozen_predecessor = self._terminal_predecessor
        if frozen_predecessor is None:
            raise RuntimeError("terminal predecessor was not frozen after planning")
        predecessor, predecessor_content_sha256 = frozen_predecessor
        current_entry = self.memory.entries_for((predecessor,))[0]
        if current_entry.draft.content_sha256 != predecessor_content_sha256:
            raise ValueError("revision predecessor content changed after reservation")
        generation_receipt_sha256s = tuple(receipt.receipt_hash for receipt in receipts)
        authority = PostEvolutionReflectionAuthority(
            spec_sha256=self.spec.spec_sha256,
            reservation_hash=context.reservation.reservation_hash,
            terminal_generation=terminal,
            generation_receipt_sha256s=generation_receipt_sha256s,
            source_scope_sha256=self.spec.source_scope.scope_sha256,
            sources=self.spec.source_scope.sources,
            source_operator_invocation_ids=tuple(
                outcome.prepared.operator_invocation_id for outcome in outcomes
            ),
            revision_predecessor=predecessor,
            revision_predecessor_content_sha256=predecessor_content_sha256,
            insight_contract_sha256=self.spec.insight_contract.identity_sha256,
            reflection_label=self.spec.label,
        )
        # Persist the authority before crossing the provider boundary so even a
        # failed logical call remains attributable to its admitted evidence.
        self.reflection_authority = authority
        try:
            result = await self.engine.reflect_with_receipt(
                outcomes,
                label=self.spec.label,
                max_insights=1,
                min_insights=0,
                insight_contract=self.spec.insight_contract,
                revision_predecessors=(predecessor,),
                source_receipt_sha256s=generation_receipt_sha256s,
            )
        except ReflectionCallExecutionError as exc:
            self.reflection_failure_type = exc.failure_type
            receipt = PostEvolutionReflectionReceipt(
                authority=authority,
                call_receipt=exc.receipt,
                reflection_status="incomplete",
                failure_type=exc.failure_type,
            )
            self.reflection_receipt = receipt
            return self._feedback_result(receipt)
        if type(result) is not ReflectionPublicationResult:
            raise TypeError("receipt-bearing reflection returned a foreign result")
        ReflectionPublicationResult.__post_init__(result)
        self.reflection_result = result
        self.reflected_entries = result.entries
        receipt = PostEvolutionReflectionReceipt(
            authority=authority,
            call_receipt=result.receipt,
            reflection_status="sealed_complete",
            failure_type=None,
        )
        self.reflection_receipt = receipt
        return self._feedback_result(receipt)

    def _feedback_result(
        self,
        receipt: PostEvolutionReflectionReceipt,
    ) -> GenerationFeedbackResult:
        metadata = [
            ("reflected_entry_count", str(len(receipt.call_receipt.publications))),
            (
                "reflection_authority_sha256",
                receipt.authority.authority_sha256,
            ),
            ("reflection_call_id", receipt.call_receipt.call_id.value),
            (
                "reflection_call_receipt_sha256",
                receipt.call_receipt.receipt_sha256,
            ),
            (
                "reflection_max_output_tokens",
                str(receipt.call_receipt.request.max_output_tokens),
            ),
            ("reflection_prompt_sha256", receipt.call_receipt.request.prompt_sha256),
            (
                "reflection_publication_outcome",
                receipt.publication_outcome,
            ),
            ("reflection_receipt_sha256", receipt.receipt_sha256),
            (
                "reflection_request_sha256",
                receipt.call_receipt.request.request_sha256,
            ),
            ("reflection_spec_sha256", self.spec.spec_sha256),
            ("reflection_status", receipt.reflection_status),
            (
                "reflection_temperature",
                (
                    "none"
                    if receipt.call_receipt.request.temperature is None
                    else float(receipt.call_receipt.request.temperature).hex()
                ),
            ),
            (
                "terminal_generation_receipt_sha256",
                receipt.authority.generation_receipt_sha256s[-1],
            ),
        ]
        if receipt.failure_type is not None:
            metadata.append(("reflection_failure_type", receipt.failure_type))
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


class PostEvolutionReflectionFactory:
    """Build the interceptor against composition-owned runtime identities."""

    def __init__(
        self,
        *,
        spec: PostEvolutionReflectionSpec,
        predecessor_resolver: PostEvolutionPredecessorResolver,
    ) -> None:
        if type(spec) is not PostEvolutionReflectionSpec:
            raise TypeError("post-evolution reflection factory spec must be exact")
        PostEvolutionReflectionSpec.__post_init__(spec)
        if not callable(predecessor_resolver):
            raise TypeError("predecessor_resolver must be callable")
        self.spec = spec
        self.predecessor_resolver = predecessor_resolver
        self.interceptor: PostEvolutionReflectionInterceptor | None = None
        self.runtime_identities: (
            tuple[object, object, object, object, object] | None
        ) = None

    def build(self, *, benchmark, engine, id_factory, memory, planner):
        if self.interceptor is not None:
            raise RuntimeError("post-evolution reflection factory is single-use")
        if type(engine) is not AgenticEvolutionEngine:
            raise TypeError("reflection factory engine must be exact")
        if type(memory) is not InsightMemoryBank:
            raise TypeError("reflection factory memory must be exact")
        if engine.ids is not id_factory or engine.memory is not memory:
            raise ValueError("reflection factory received foreign engine identities")
        problem = getattr(benchmark, "problem", None)
        if problem is not None and engine.problem is not problem:
            raise ValueError("reflection factory benchmark differs from engine")
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("reflection factory planner must implement plan")
        for attribute, expected in (
            ("benchmark", benchmark),
            ("engine", engine),
            ("ids", id_factory),
            ("id_factory", id_factory),
            ("memory", memory),
        ):
            if (
                hasattr(planner, attribute)
                and getattr(planner, attribute) is not expected
            ):
                raise ValueError(
                    f"reflection factory planner has a foreign {attribute} identity"
                )
        self.runtime_identities = (
            benchmark,
            engine,
            id_factory,
            memory,
            planner,
        )
        self.interceptor = PostEvolutionReflectionInterceptor(
            engine=engine,
            planner=planner,
            memory=memory,
            spec=self.spec,
            predecessor_resolver=self.predecessor_resolver,
        )
        return self.interceptor


__all__ = [
    "POST_EVOLUTION_REFLECTION_DEFINITION_SHA256",
    "PostEvolutionPredecessorResolver",
    "PostEvolutionReflectionAuthority",
    "PostEvolutionReflectionFactory",
    "PostEvolutionReflectionInterceptor",
    "PostEvolutionReflectionReceipt",
    "PostEvolutionReflectionSource",
    "PostEvolutionReflectionSourceScope",
    "PostEvolutionReflectionSpec",
]
