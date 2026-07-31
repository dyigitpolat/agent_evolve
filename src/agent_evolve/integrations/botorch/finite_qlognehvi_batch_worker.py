"""Isolated JSON worker for common-realization qLogNEHVI slate scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_evolve.integrations.botorch.finite_qlognehvi_batch import (
    BotorchQLogNehviFiniteBatchScorePolicy,
)
from agent_evolve.ports.finite_acquisition_batch_json import (
    finite_acquisition_batch_score_request_from_record,
)


WORKER_SCHEMA_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--mc-samples", required=True, type=int)
    parser.add_argument("--maximum-score-batch-size", required=True, type=int)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    payload = json.loads(arguments.request.read_text(encoding="ascii"))
    request = finite_acquisition_batch_score_request_from_record(payload)
    policy = BotorchQLogNehviFiniteBatchScorePolicy(
        mc_samples=arguments.mc_samples,
        maximum_score_batch_size=arguments.maximum_score_batch_size,
    )
    decision = policy.score(request)
    response = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "decision": decision.to_record(),
    }
    arguments.response.write_text(
        json.dumps(
            response,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="ascii",
    )


if __name__ == "__main__":  # pragma: no cover - exercised as a child process
    main()
