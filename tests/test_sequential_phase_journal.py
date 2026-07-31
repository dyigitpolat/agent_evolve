from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from agent_evolve.application.sequential_residual_portfolio_evolution import (
    SequentialResidualPhase,
    SequentialResidualPhaseReceipt,
)
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.infrastructure.sequential_phase_journal import (
    DurableJsonlSequentialPhaseCommitter,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _receipt() -> SequentialResidualPhaseReceipt:
    return SequentialResidualPhaseReceipt(
        phase=SequentialResidualPhase.PLAN_FROZEN,
        residual_request_sha256=_sha("request"),
        plan_sha256=_sha("plan"),
        product_sha256s=(_sha("locked"), _sha("unlocked")),
        evidence=freeze_json(
            {
                "allocation_plan": {
                    "pilot": ["pilot-action"],
                    "locked": ["pilot-action", "locked-action"],
                    "unlocked": ["pilot-action", "recursive-action"],
                }
            }
        ),
    )


def test_phase_journal_fsyncs_full_receipt_and_rejects_duplicates(
    tmp_path,
) -> None:
    path = tmp_path / "sequential_phase_receipts.jsonl"
    committer = DurableJsonlSequentialPhaseCommitter(path)
    receipt = _receipt()

    ack = asyncio.run(committer.commit(receipt))

    assert ack.durable
    row = json.loads(path.read_text(encoding="ascii"))
    assert row["receipt"]["receipt_sha256"] == receipt.receipt_sha256
    assert row["receipt"]["evidence"]["allocation_plan"]["pilot"] == [
        "pilot-action"
    ]
    assert row["ack"]["durable"] is True
    reloaded = DurableJsonlSequentialPhaseCommitter(path)
    with pytest.raises(ValueError, match="already durably committed"):
        asyncio.run(reloaded.commit(receipt))
