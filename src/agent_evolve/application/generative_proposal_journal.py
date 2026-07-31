"""The durable side of a generative seal: one chained JSONL per campaign.

One line per model call, in issue order, each carrying the digest of the line
before it. Reading it back reconstructs the exact call sequence and re-derives
every digest, so a journal that has been edited, reordered, truncated at the
front, or had a call inserted into it fails to close.

The last property is the one that matters most here. An agent on this project
filled unevaluated compositions with predicted values and reported a 1.7x win
that had to be retracted. A chained journal makes the analogous move on the
proposal side impossible to do quietly: a call that did not happen has no
predecessor digest to inherit, and the terminal digest of the campaign is
published with the result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent_evolve.domain.generative_emission import (
    GenerativeEmission,
    GenerativeProposalCall,
    SealedGuidanceCall,
    SealedRunHeader,
    chain_sealed_calls,
)
from agent_evolve.domain.typed_json import freeze_json

__all__ = [
    "candidate_schema_sha256",
    "journal_line",
    "read_generative_journal",
    "verify_generative_journal",
    "write_generative_journal",
]

_SCHEMA_HASH_DOMAIN = b"agent-evolve:candidate-schema-identity:v1\x00"


def candidate_schema_sha256(candidate_model: Any) -> str:
    """Digest the exact JSON schema a proposal must satisfy.

    This is the support of the generative operator, and therefore the support a
    matched null has to sample. Recording it turns "the null draws from the same
    space" from a claim in a write-up into a field two runs can be compared on.
    """

    if candidate_model is None:
        raise ValueError(
            "a generative campaign needs a candidate schema. Without one the "
            "proposer has no declared support and no null can be matched to it."
        )
    schema = candidate_model.model_json_schema()
    payload = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(_SCHEMA_HASH_DOMAIN + payload).hexdigest()


def journal_line(record: dict) -> str:
    """Serialise one sealed call as canonical JSON on a single line."""

    return json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def write_generative_journal(path: Path, calls: Sequence[Any]) -> str:
    """Write the chain and return its terminal digest."""

    terminal = chain_sealed_calls(tuple(calls))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for call in calls:
            handle.write(journal_line(call.to_record()) + "\n")
    return terminal


def _call_from_record(record: dict) -> Any:
    if record.get("schema_version") != 1:
        raise ValueError("unsupported sealed call schema_version")
    kind = record.get("record_kind")
    if kind == "run_header":
        call = SealedRunHeader(
            proposer_id=str(record["proposer_id"]),
            requested_model=str(record["requested_model"]),
            candidate_schema_sha256=str(record["candidate_schema_sha256"]),
            provides_insights=bool(record["provides_insights"]),
        )
    elif "emissions" in record:
        emissions = tuple(
            GenerativeEmission(
                configuration=freeze_json(dict(item["configuration"])),
                accepted=bool(item["accepted"]),
                rejection_reason=str(item.get("rejection_reason", "")),
            )
            for item in record["emissions"]
        )
        call = GenerativeProposalCall(
            call_ordinal=int(record["call_ordinal"]),
            op=str(record["op"]),
            requested_model=str(record["requested_model"]),
            prompt_sha256=str(record["prompt_sha256"]),
            candidate_schema_sha256=str(record["candidate_schema_sha256"]),
            emissions=emissions,
            previous_call_sha256=str(record["previous_call_sha256"]),
        )
    elif "outputs" in record:
        call = SealedGuidanceCall(
            call_ordinal=int(record["call_ordinal"]),
            op=str(record["op"]),
            requested_model=str(record["requested_model"]),
            prompt_sha256=str(record["prompt_sha256"]),
            outputs=tuple(str(x) for x in record["outputs"]),
            previous_call_sha256=str(record["previous_call_sha256"]),
        )
    else:
        raise ValueError("a sealed call carries either emissions or outputs")
    if call.identity_sha256 != record["call_identity_sha256"]:
        raise ValueError(
            f"call {record['call_ordinal']} does not authenticate: its recorded "
            "identity is not the identity of its contents"
        )
    return call


def read_generative_journal(path: Path) -> tuple:
    """Read a journal back into sealed call objects, authenticating each line."""

    calls = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.strip():
                continue
            calls.append(_call_from_record(json.loads(line)))
    return tuple(calls)


def verify_generative_journal(path: Path) -> dict:
    """Authenticate a journal end to end and summarise what it says happened.

    Every count here is derived from the sealed content. Nothing is declared.
    """

    calls = read_generative_journal(path)
    terminal = chain_sealed_calls(calls)
    header = calls[0]
    proposals = tuple(c for c in calls if type(c) is GenerativeProposalCall)
    emitted = sum(len(c.emissions) for c in proposals)
    accepted = sum(1 for c in proposals for e in c.emissions if e.accepted)
    distinct = {
        e.configuration_sha256 for c in proposals for e in c.emissions if e.accepted
    }
    schemas = {c.candidate_schema_sha256 for c in proposals}
    models = {c.requested_model for c in calls}
    return {
        "path": str(path),
        "terminal_sha256": terminal,
        "proposer_id": header.proposer_id,
        "provides_insights": header.provides_insights,
        "calls": len(calls) - 1,
        "proposal_calls": len(proposals),
        "guidance_calls": len(calls) - 1 - len(proposals),
        "emitted_configurations": emitted,
        "accepted_configurations": accepted,
        "distinct_accepted_configurations": len(distinct),
        "candidate_schema_sha256s": sorted(schemas),
        "requested_models": sorted(models),
    }


def iter_accepted_configurations(calls: Iterable[Any]) -> tuple:
    """Every configuration the model authored that ``validate`` let through."""

    out = []
    for call in calls:
        if type(call) is not GenerativeProposalCall:
            continue
        for emission in call.emissions:
            if emission.accepted:
                out.append(emission.configuration)
    return tuple(out)
