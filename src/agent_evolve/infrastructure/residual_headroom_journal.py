"""Durable JSONL compare-and-swap store for residual-headroom closures."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from agent_evolve.application.residual_headroom_campaign_runtime import (
    ResidualHeadroomLedgerCommitAck,
)
from agent_evolve.application.residual_headroom_ledger import (
    ConservedResidualHeadroomLedger,
    ResidualHeadroomLedgerState,
    ResidualHeadroomStageClosure,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json


DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_ID = (
    "durable_jsonl_residual_headroom_store"
)
DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_VERSION = 1
DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:durable-jsonl-residual-headroom-store:v1;"
    b"compare-and-swap=expected-prior-state-sha256;"
    b"record=full-authenticated-closure-plus-transition-ack;"
    b"write=append-flush-fsync-before-state-publication;"
    b"recovery=replay-and-authenticate-every-transition;"
    b"partial-or-foreign-record=fail-closed"
).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ack_from_record(
    record: dict[str, object],
) -> ResidualHeadroomLedgerCommitAck:
    if type(record) is not dict:
        raise TypeError("commit ack record must be an exact object")
    store = record["store"]
    if type(store) is not dict:
        raise TypeError("commit ack store identity must be an object")
    evidence = record.get("evidence")
    if type(evidence) is not dict:
        raise TypeError("durable commit ack must contain its evidence")
    value = ResidualHeadroomLedgerCommitAck(
        store_id=str(store["store_id"]),
        store_version=int(store["store_version"]),
        store_definition_sha256=str(store["definition_sha256"]),
        prior_state_sha256=str(record["prior_state_sha256"]),
        closure_sha256=str(record["closure_sha256"]),
        new_state_sha256=str(record["new_state_sha256"]),
        durable=bool(record["durable"]),
        evidence=freeze_json(evidence),
    )
    if value.ack_sha256 != str(record["ack_sha256"]):
        raise ValueError("commit ack record hash does not authenticate")
    return value


@dataclass(slots=True)
class DurableJsonlResidualHeadroomStore:
    """Append, fsync, and recover every conserved headroom transition."""

    path: Path
    ledger: ConservedResidualHeadroomLedger
    store_id: str = DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_ID
    store_version: int = DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_VERSION
    definition_sha256: str = (
        DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256
    )
    state: ResidualHeadroomLedgerState = field(init=False)
    commit_acks: tuple[ResidualHeadroomLedgerCommitAck, ...] = field(
        init=False,
        default=(),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a pathlib.Path")
        if type(self.ledger) is not ConservedResidualHeadroomLedger:
            raise TypeError("ledger must be exact")
        self.ledger.__post_init__()
        require_sha256(self.definition_sha256, "definition_sha256")
        if (
            self.store_id != DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_ID
            or self.store_version
            != DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_VERSION
            or self.definition_sha256
            != DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256
        ):
            raise ValueError("durable store identity is immutable")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = self.ledger.empty_state()
        acknowledgements: list[ResidualHeadroomLedgerCommitAck] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"headroom journal line {line_number} is partial"
                        ) from error
                    if type(record) is not dict:
                        raise TypeError(
                            "headroom journal entries must be exact objects"
                        )
                    if int(record.get("schema_version", 0)) != 1:
                        raise ValueError(
                            "headroom journal schema is unsupported"
                        )
                    if (
                        str(record["prior_state_sha256"])
                        != state.state_sha256
                    ):
                        raise ValueError(
                            "headroom journal state chain does not close"
                        )
                    closure_record = record["closure"]
                    ack_record = record["commit_ack"]
                    if (
                        type(closure_record) is not dict
                        or type(ack_record) is not dict
                    ):
                        raise TypeError(
                            "journal closure and ack must be objects"
                        )
                    closure = ResidualHeadroomStageClosure.from_record(
                        closure_record
                    )
                    new_state = self.ledger.append(state, closure)
                    ack = _ack_from_record(ack_record)
                    if (
                        ack.store_id != self.store_id
                        or ack.store_version != self.store_version
                        or ack.store_definition_sha256
                        != self.definition_sha256
                        or ack.prior_state_sha256 != state.state_sha256
                        or ack.closure_sha256 != closure.closure_sha256
                        or ack.new_state_sha256
                        != new_state.state_sha256
                        or not ack.durable
                        or str(record["new_state_sha256"])
                        != new_state.state_sha256
                    ):
                        raise ValueError(
                            "headroom journal transition does not authenticate"
                        )
                    state = new_state
                    acknowledgements.append(ack)
        self.state = state
        self.commit_acks = tuple(acknowledgements)

    async def commit(
        self,
        *,
        expected_prior_state_sha256: str,
        closure: ResidualHeadroomStageClosure,
    ) -> ResidualHeadroomLedgerCommitAck:
        require_sha256(
            expected_prior_state_sha256,
            "expected_prior_state_sha256",
        )
        if expected_prior_state_sha256 != self.state.state_sha256:
            raise ValueError("headroom state compare-and-swap failed")
        if type(closure) is not ResidualHeadroomStageClosure:
            raise TypeError("closure must be exact")
        closure.__post_init__()
        new_state = self.ledger.append(self.state, closure)
        ack = ResidualHeadroomLedgerCommitAck(
            store_id=self.store_id,
            store_version=self.store_version,
            store_definition_sha256=self.definition_sha256,
            prior_state_sha256=self.state.state_sha256,
            closure_sha256=closure.closure_sha256,
            new_state_sha256=new_state.state_sha256,
            durable=True,
            evidence=freeze_json(
                {
                    "storage": "append_only_jsonl",
                    "flush": True,
                    "fsync": True,
                    "sequence": len(self.commit_acks) + 1,
                }
            ),
        )
        record = {
            "schema_version": 1,
            "prior_state_sha256": self.state.state_sha256,
            "closure": closure.to_record(),
            "new_state_sha256": new_state.state_sha256,
            "commit_ack": ack.to_record(include_evidence=True),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.state = new_state
        self.commit_acks = (*self.commit_acks, ack)
        return ack


__all__ = [
    "DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_DEFINITION_SHA256",
    "DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_ID",
    "DURABLE_JSONL_RESIDUAL_HEADROOM_STORE_VERSION",
    "DurableJsonlResidualHeadroomStore",
]
