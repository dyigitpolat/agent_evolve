"""Sealed *model-authored* candidates: replay by content, not by menu index.

The finite variation catalogue (:mod:`agent_evolve.domain.finite_variation`)
seals a decision as an **opaque option identifier** and lets trusted benchmark
code own every child configuration.  That makes replay trivial -- resolve the id
against the sealed contract -- at the cost of deleting the operator: the model
never authors a genotype, it filters a list.

This module is the other half of that trade.  A generative proposal seals the
**emitted configuration itself**, together with the exact prompt that produced
it and the schema that bounded it, so a replay reconstructs the decision from
the emission rather than from a catalogue index.  Nothing here needs a
provider, a network, or a pre-enumerated option set.

Three properties are preserved from the catalogue scheme, and one is
deliberately given up.

Preserved:

``provider-free replay``   the configuration is stored verbatim as canonical
                           typed JSON, so replay is a parse, not a call.
``content addressing``     every record self-authenticates under a separated
                           hash domain, and calls are chained, so a call that
                           did not happen cannot be inserted after the fact.
``support pinning``        ``candidate_schema_sha256`` records the exact schema
                           the emission was drawn from, which is what makes a
                           matched null checkable: the null must sample *that*
                           schema, not some other one.

Given up, on purpose:

``bounded model authority``  the model now authors configurations. That is the
                             mechanism the catalogue was suppressing, so the
                             replacement guarantee is feasibility rather than
                             enumeration -- every emission carries the verdict
                             the problem's own ``validate`` returned for it, and
                             replay re-runs ``validate`` and requires agreement.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)

__all__ = [
    "GENESIS_CALL_SHA256",
    "GenerativeEmission",
    "GenerativeProposalCall",
    "MAX_EMISSIONS_PER_CALL",
    "SealedGuidanceCall",
    "chain_sealed_calls",
    "generative_prompt_sha256",
    "validate_generative_emission",
    "validate_generative_proposal_call",
    "validate_sealed_guidance_call",
]


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_MAX_REASON_BYTES = 4_096

#: An emitted batch is a proposal, not a corpus. The loop asks for at most a
#: generation's worth; anything larger is a runaway model, not a decision.
MAX_EMISSIONS_PER_CALL = 256

#: The predecessor digest of the first call in a campaign. A chain that starts
#: anywhere else is a chain with a missing head.
GENESIS_CALL_SHA256 = "0" * 64

#: Guidance text that survives into the next prompt. A model that writes a
#: paragraph of insight and then reads it back next generation is running a
#: feedback loop, and replaying the proposals without it would reconstruct a
#: different question. Bounded so a runaway generation cannot be sealed.
MAX_GUIDANCE_BYTES = 262_144

_PROMPT_HASH_DOMAIN = b"agent-evolve:generative-proposal-prompt:v1\x00"
_EMISSION_HASH_DOMAIN = b"agent-evolve:generative-proposal-emission:v1\x00"
_CALL_HASH_DOMAIN = b"agent-evolve:generative-proposal-call:v1\x00"
_GUIDANCE_CALL_HASH_DOMAIN = b"agent-evolve:generative-guidance-call:v1\x00"


def _frame(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("framed payloads must be exact bytes")
    return len(payload).to_bytes(8, "big", signed=False) + payload


def _bounded_reason(value: str) -> bytes:
    if type(value) is not str:
        raise TypeError("rejection_reason must be an exact string")
    encoded = value.encode("utf-8", errors="strict")
    if len(encoded) > _MAX_REASON_BYTES:
        raise ValueError("rejection_reason exceeds its byte limit")
    return encoded


def generative_prompt_sha256(instruction: str) -> str:
    """Digest the exact instruction text a generative call was issued with.

    The prompt is hashed rather than stored: it is reconstructible from the
    sealed campaign inputs, and a hash is what a replay needs in order to prove
    it is answering the same question.  Domain separation keeps this digest from
    colliding with any other use of the same bytes.
    """

    if type(instruction) is not str:
        raise TypeError("instruction must be an exact string")
    if not instruction:
        raise ValueError("instruction must be non-empty")
    return hashlib.sha256(
        _PROMPT_HASH_DOMAIN + _frame(instruction.encode("utf-8", errors="strict"))
    ).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class GenerativeEmission:
    """One configuration the model authored, with the verdict it earned.

    ``configuration`` is the complete candidate, frozen typed JSON, exactly as
    it will be handed to ``materialize``.  ``accepted`` is what the problem's
    ``validate`` said about it at emission time; a replay that re-validates and
    disagrees is a broken replay, not a repaired one.
    """

    configuration: FrozenJsonObject
    accepted: bool
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if type(self.configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be an exact FrozenJsonObject")
        if freeze_json(self.configuration) is not self.configuration:
            raise TypeError("configuration must already be frozen typed JSON")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be an exact bool")
        _bounded_reason(self.rejection_reason)
        if self.accepted and self.rejection_reason:
            raise ValueError("an accepted emission carries no rejection reason")
        if not self.accepted and not self.rejection_reason:
            raise ValueError(
                "a rejected emission must carry the reason validate() gave, "
                "because that reason is what the proposer was shown"
            )

    @property
    def configuration_sha256(self) -> str:
        validate_generative_emission(self)
        return typed_json_sha256(self.configuration)

    @property
    def identity_sha256(self) -> str:
        """Bind the full configuration and its feasibility verdict."""

        validate_generative_emission(self)
        digest = hashlib.sha256()
        digest.update(_EMISSION_HASH_DOMAIN)
        digest.update(_frame(canonical_typed_json_bytes(self.configuration)))
        digest.update(b"\x01" if self.accepted else b"\x00")
        digest.update(_frame(_bounded_reason(self.rejection_reason)))
        return digest.hexdigest()

    def to_record(self) -> dict[str, object]:
        """Return the durable projection. The configuration is *not* elided.

        A catalogue seal can publish an option id and stay content-free because
        the contract holds the child.  Here the emission is the only copy, so
        replay needs it in full.
        """

        validate_generative_emission(self)
        return {
            "configuration": _thawed(self.configuration),
            "configuration_sha256": self.configuration_sha256,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "emission_identity_sha256": self.identity_sha256,
        }

    def _validated_values(self) -> tuple[object, ...]:
        validate_generative_emission(self)
        return (
            canonical_typed_json_bytes(self.configuration),
            self.accepted,
            self.rejection_reason,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not GenerativeEmission or type(other) is not GenerativeEmission:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((GenerativeEmission, self._validated_values()))


def _thawed(value: FrozenJsonObject) -> dict[str, object]:
    from agent_evolve.domain.typed_json import thaw_json

    thawed = thaw_json(value)
    if type(thawed) is not dict:
        raise TypeError("a frozen JSON object must thaw to an exact dict")
    return thawed


def validate_generative_emission(emission: GenerativeEmission) -> None:
    """Revalidate an exact emission at a public trust boundary."""

    if type(emission) is not GenerativeEmission:
        raise TypeError("emission must be an exact GenerativeEmission")
    GenerativeEmission.__post_init__(emission)


@dataclass(frozen=True, slots=True, eq=False)
class GenerativeProposalCall:
    """One sealed model call: what was asked, what came back, and after what.

    ``previous_call_sha256`` chains the calls of a campaign in issue order.  The
    chain is the part that makes fabrication detectable: a record invented for a
    call that never happened cannot be spliced in without recomputing every
    digest after it, and the campaign's terminal digest is published.
    """

    call_ordinal: int
    op: str
    requested_model: str
    prompt_sha256: str
    candidate_schema_sha256: str
    emissions: tuple[GenerativeEmission, ...]
    previous_call_sha256: str = GENESIS_CALL_SHA256

    def __post_init__(self) -> None:
        if type(self.call_ordinal) is not int or self.call_ordinal < 0:
            raise ValueError("call_ordinal must be a non-negative exact integer")
        if type(self.op) is not str or _TOKEN.fullmatch(self.op) is None:
            raise ValueError("op must use the closed lowercase token grammar")
        if (
            type(self.requested_model) is not str
            or _MODEL_ID.fullmatch(self.requested_model) is None
        ):
            raise ValueError("requested_model must be a closed model identifier")
        require_sha256(self.prompt_sha256, "prompt_sha256")
        require_sha256(self.candidate_schema_sha256, "candidate_schema_sha256")
        require_sha256(self.previous_call_sha256, "previous_call_sha256")
        if type(self.emissions) is not tuple:
            raise TypeError("emissions must be an exact tuple")
        if not self.emissions:
            raise ValueError(
                "a sealed call must carry at least one emission; a call that "
                "returned nothing is a failure to record as a failure, not an "
                "empty success"
            )
        if len(self.emissions) > MAX_EMISSIONS_PER_CALL:
            raise ValueError("emission batch exceeds its limit")
        for emission in self.emissions:
            validate_generative_emission(emission)

    @property
    def identity_sha256(self) -> str:
        """Bind the request identity, every emission, and the predecessor."""

        validate_generative_proposal_call(self)
        digest = hashlib.sha256()
        digest.update(_CALL_HASH_DOMAIN)
        digest.update(bytes.fromhex(self.previous_call_sha256))
        digest.update(self.call_ordinal.to_bytes(8, "big", signed=False))
        digest.update(_frame(self.op.encode("ascii")))
        digest.update(_frame(self.requested_model.encode("ascii")))
        digest.update(bytes.fromhex(self.prompt_sha256))
        digest.update(bytes.fromhex(self.candidate_schema_sha256))
        digest.update(len(self.emissions).to_bytes(8, "big", signed=False))
        for emission in self.emissions:
            digest.update(bytes.fromhex(emission.identity_sha256))
        return digest.hexdigest()

    @property
    def accepted_configurations(self) -> tuple[FrozenJsonObject, ...]:
        validate_generative_proposal_call(self)
        return tuple(e.configuration for e in self.emissions if e.accepted)

    def to_record(self) -> dict[str, object]:
        validate_generative_proposal_call(self)
        return {
            "schema_version": 1,
            "call_ordinal": self.call_ordinal,
            "op": self.op,
            "requested_model": self.requested_model,
            "prompt_sha256": self.prompt_sha256,
            "candidate_schema_sha256": self.candidate_schema_sha256,
            "previous_call_sha256": self.previous_call_sha256,
            "emissions": [e.to_record() for e in self.emissions],
            "call_identity_sha256": self.identity_sha256,
        }

    def _validated_values(self) -> tuple[object, ...]:
        validate_generative_proposal_call(self)
        return (
            self.call_ordinal,
            self.op,
            self.requested_model,
            self.prompt_sha256,
            self.candidate_schema_sha256,
            self.previous_call_sha256,
            tuple(e._validated_values() for e in self.emissions),
        )

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not GenerativeProposalCall
            or type(other) is not GenerativeProposalCall
        ):
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((GenerativeProposalCall, self._validated_values()))


def validate_generative_proposal_call(call: GenerativeProposalCall) -> None:
    """Revalidate an exact call at a public trust boundary."""

    if type(call) is not GenerativeProposalCall:
        raise TypeError("call must be an exact GenerativeProposalCall")
    GenerativeProposalCall.__post_init__(call)


@dataclass(frozen=True, slots=True, eq=False)
class SealedGuidanceCall:
    """One sealed model call that returned text rather than configurations.

    Insight and constraint-guide calls author no candidate, so they are not the
    operator under test -- but their output is *pasted into the next prompt*.
    Leaving them out of the chain would make the next proposal call unreplayable
    for a reason that looks like drift and is really an omission, so they are
    sealed at the same altitude and in the same sequence.
    """

    call_ordinal: int
    op: str
    requested_model: str
    prompt_sha256: str
    outputs: tuple[str, ...]
    previous_call_sha256: str = GENESIS_CALL_SHA256

    def __post_init__(self) -> None:
        if type(self.call_ordinal) is not int or self.call_ordinal < 0:
            raise ValueError("call_ordinal must be a non-negative exact integer")
        if type(self.op) is not str or _TOKEN.fullmatch(self.op) is None:
            raise ValueError("op must use the closed lowercase token grammar")
        if (
            type(self.requested_model) is not str
            or _MODEL_ID.fullmatch(self.requested_model) is None
        ):
            raise ValueError("requested_model must be a closed model identifier")
        require_sha256(self.prompt_sha256, "prompt_sha256")
        require_sha256(self.previous_call_sha256, "previous_call_sha256")
        if type(self.outputs) is not tuple:
            raise TypeError("outputs must be an exact tuple")
        total = 0
        for value in self.outputs:
            if type(value) is not str:
                raise TypeError("every guidance output must be an exact string")
            total += len(value.encode("utf-8", errors="strict"))
        if total > MAX_GUIDANCE_BYTES:
            raise ValueError("guidance output exceeds its byte limit")

    @property
    def identity_sha256(self) -> str:
        validate_sealed_guidance_call(self)
        digest = hashlib.sha256()
        digest.update(_GUIDANCE_CALL_HASH_DOMAIN)
        digest.update(bytes.fromhex(self.previous_call_sha256))
        digest.update(self.call_ordinal.to_bytes(8, "big", signed=False))
        digest.update(_frame(self.op.encode("ascii")))
        digest.update(_frame(self.requested_model.encode("ascii")))
        digest.update(bytes.fromhex(self.prompt_sha256))
        digest.update(len(self.outputs).to_bytes(8, "big", signed=False))
        for value in self.outputs:
            digest.update(_frame(value.encode("utf-8", errors="strict")))
        return digest.hexdigest()

    def to_record(self) -> dict[str, object]:
        validate_sealed_guidance_call(self)
        return {
            "schema_version": 1,
            "call_ordinal": self.call_ordinal,
            "op": self.op,
            "requested_model": self.requested_model,
            "prompt_sha256": self.prompt_sha256,
            "previous_call_sha256": self.previous_call_sha256,
            "outputs": list(self.outputs),
            "call_identity_sha256": self.identity_sha256,
        }

    def _validated_values(self) -> tuple[object, ...]:
        validate_sealed_guidance_call(self)
        return (
            self.call_ordinal,
            self.op,
            self.requested_model,
            self.prompt_sha256,
            self.previous_call_sha256,
            self.outputs,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not SealedGuidanceCall or type(other) is not SealedGuidanceCall:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((SealedGuidanceCall, self._validated_values()))


def validate_sealed_guidance_call(call: SealedGuidanceCall) -> None:
    """Revalidate an exact guidance call at a public trust boundary."""

    if type(call) is not SealedGuidanceCall:
        raise TypeError("call must be an exact SealedGuidanceCall")
    SealedGuidanceCall.__post_init__(call)


def chain_sealed_calls(
    calls: tuple[object, ...],
) -> str:
    """Verify a campaign's call chain and return its terminal digest.

    Raises on the three ways a chain can lie: a missing head, a gap or a
    reordering in the ordinals, and a predecessor digest that does not match the
    call it claims to follow. Proposal and guidance calls share one sequence
    because they share one conversation, and their hash domains differ so one
    can never be replayed as the other.
    """

    if type(calls) is not tuple:
        raise TypeError("calls must be an exact tuple")
    if not calls:
        raise ValueError("a campaign chain requires at least one call")
    previous = GENESIS_CALL_SHA256
    for index, call in enumerate(calls):
        if type(call) is GenerativeProposalCall:
            validate_generative_proposal_call(call)
        elif type(call) is SealedGuidanceCall:
            validate_sealed_guidance_call(call)
        else:
            raise TypeError("a sealed chain holds only exact sealed call records")
        if call.call_ordinal != index:
            raise ValueError(
                f"call chain is not contiguous at position {index}: "
                f"ordinal {call.call_ordinal}"
            )
        if call.previous_call_sha256 != previous:
            raise ValueError(
                f"call {index} does not follow the call it claims to follow"
            )
        previous = call.identity_sha256
    return previous
