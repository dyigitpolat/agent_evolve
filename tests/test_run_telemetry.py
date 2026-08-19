"""Guidance telemetry must leave the run inside the ``SearchResult``.

The seam objects have counted their own behaviour since they existed
(``ChooserTelemetry``, ``PriorTelemetry``) — but the counters lived as
function attributes on closures the caller discarded, so no run could report
what its guidance actually did. These tests pin the fix at the public surface:
a result always carries a measured telemetry block, and a guided run's
mechanism counters are reachable from it.

``test_the_genetic_llm_run_consults_the_chooser`` is also the regression test
for a live wiring bug: ``optimize()`` built the LLM chooser and never passed
it to the loop, so ``proposer='llm'`` ran byte-identical to ``'random'`` while
looking guided. On the pre-fix tree this test fails.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Sequence

from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Seeded:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec(name="ones", goal="max"),)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return {"ones": float(sum(artifact))}


def test_an_offline_run_reports_measured_telemetry() -> None:
    result = optimize(_Seeded(), budget=12, seed=3)
    assert result.telemetry is not None, "the telemetry block is absent"
    assert result.telemetry.real_evaluations == result.evaluations
    assert result.telemetry.virtual_evaluations == 0
    assert not [m for m in result.telemetry.mechanisms if m.authored_by == "llm"], (
        "an offline run reported an llm-authored mechanism"
    )


def test_an_offline_run_reports_a_measured_zero_provider_usage() -> None:
    # Zero calls is a measurement, not an omission: an absent block cannot be
    # told apart from an unrecorded one.
    result = optimize(_Seeded(), budget=8, seed=4)
    assert result.provider_usage is not None
    assert result.provider_usage.calls == 0
    assert result.provider_usage.provider_free


def test_the_genetic_llm_run_consults_the_chooser(monkeypatch) -> None:
    # A canned completion stands in for the provider; what is under test is the
    # wiring: the chooser must be constructed AND handed to the loop, and its
    # counters must be reachable from the result.
    reply = (
        '[{"parent_a": 0, "parent_b": 1, "mask": [0, 1, 0, 1, 0, 1]},'
        ' {"parent_a": 1, "parent_b": 2, "mask": [1, 1, 1, 0, 0, 0]}]'
    )

    def _canned_completion_for(model, settings=None, **kwargs):
        return lambda prompt: reply

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        _canned_completion_for,
    )
    # 2026-08-19: the per-offspring chooser became opt-in (`chooser="llm"`).
    # Ten sealed null verdicts at 107-171x the cost of the run it advises, so
    # a model run no longer buys it by default; the wiring under test here is
    # unchanged, and asking for it by name is now how it is reached.
    result = optimize(_Seeded(), budget=16, seed=5, proposer="llm",
                      chooser="llm")

    assert result.telemetry is not None
    rows = [m for m in result.telemetry.mechanisms if m.mechanism == "chooser"]
    assert rows, (
        "no chooser telemetry reached the result: the llm chooser was built "
        "and dropped (the pre-fix wiring), or harvesting is broken"
    )
    assert rows[0].authored_by == "llm"
    assert rows[0].counters["calls"] > 0, "the chooser was never consulted"
    assert rows[0].counters["accepted"] > 0, "no canned choice was accepted"
