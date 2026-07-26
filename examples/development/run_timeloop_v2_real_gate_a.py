#!/usr/bin/env python3
"""Provider-free six-generation Timeloop campaign over the real evaluator.

This is a release/qualification run, not an AgentEvolve efficacy claim.  It
uses the exact campaign orchestration exercised by the cheap structural double,
while replacing its objective function with the pinned Docker evaluator and
the workload-owned detailed-evidence adapter.  No model provider is reachable
from this executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json  # noqa: E402
from agent_evolve.infrastructure.artifacts.filesystem import (  # noqa: E402
    FileSystemArtifactStore,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (  # noqa: E402
    compose_timeloop_v2_detailed_benchmark,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    TimeloopV2DockerEvaluator,
    TimeloopV2Evaluation,
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.development.run_timeloop_v2_provider_free_campaign import (  # noqa: E402
    run_timeloop_campaign,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("Timeloop real Gate A receipt is not an object")
    return frozen


class _CountingEvaluator:
    """Count physical Docker calls without changing evaluator semantics."""

    def __init__(self, delegate: TimeloopV2DockerEvaluator) -> None:
        self.delegate = delegate
        self.calls = 0

    def evaluate(self, config: object) -> TimeloopV2Evaluation:
        self.calls += 1
        return self.delegate.evaluate(config)


def run_real_timeloop_gate_a(output_root: Path):
    """Run the fixed G6 qualification campaign with no provider.

    The schedule contains 38 candidate occurrences; the exact-phenotype cache
    may require fewer physical Docker calls when a recombination repeats an
    already evaluated configuration.  Both counts are retained separately.
    """

    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    output_root.mkdir(parents=True, exist_ok=True)
    panel = frozen_network_panel("resnet50")
    settings = TimeloopV2Settings(
        output_root=output_root / "evaluator_calls",
        cpu_set="8",
        timeout_s=180.0,
    )
    raw_evaluator = TimeloopV2DockerEvaluator(settings, panel)
    preflight = raw_evaluator.preflight()
    evaluator = _CountingEvaluator(raw_evaluator)
    artifact_store = FileSystemArtifactStore(output_root / "artifact_store")
    benchmark = compose_timeloop_v2_detailed_benchmark(
        settings,
        panel,
        artifact_store=artifact_store,
        evaluator=evaluator,
    )
    detailed = benchmark.detailed_evaluator
    if detailed is None:  # pragma: no cover - composition is fail-closed.
        raise AssertionError("real Timeloop benchmark omitted detailed evidence")
    run = run_timeloop_campaign(
        benchmark=benchmark,
        evaluator=evaluator,
        execution_mode="real_docker_provider_free_gate_a",
        id_namespace="timeloop_v2_real_gate_a_20260717",
        campaign_sha256=_sha("timeloop-v2-real-gate-a-campaign-v1"),
        evaluator_contract_sha256=(
            detailed.evaluator_identity.evaluator_context_sha256
        ),
        protocol_id="timeloop_v2_real_gate_a",
        protocol_definition_sha256=_sha("timeloop-v2-real-gate-a-protocol-v1"),
        task_sha256=_sha("timeloop-v2-resnet50-real-gate-a-task-v1"),
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "mode": "real_docker_provider_free_gate_a",
                "preflight": preflight,
            }
        ),
        resource_lease_receipt=_object(
            {
                "resource": "serial_timeloop_docker_cpu_8",
                "active": True,
                "evaluator_concurrency": 1,
            }
        ),
        docker_enabled=True,
        scientific_claim="real_evaluator_structural_conformance_only",
    )
    summary = run.summary()
    summary.update(
        {
            "artifact_store_file_count": len(
                tuple((output_root / "artifact_store").glob("*.artifact"))
            ),
            "evaluation_directory_count": len(
                tuple((output_root / "evaluator_calls").glob("timeloop-v2-*"))
            ),
            "evaluator_context_sha256": (
                detailed.evaluator_identity.evaluator_context_sha256
            ),
            "output_root": str(output_root.resolve()),
        }
    )
    (output_root / "campaign_journal.json").write_text(
        json.dumps(
            [event.to_record() for event in run.journal.events],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "gate_a_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _, summary = run_real_timeloop_gate_a(args.output_root)
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
