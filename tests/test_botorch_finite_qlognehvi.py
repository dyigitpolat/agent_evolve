from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("botorch")

from agent_evolve.integrations.botorch import (  # noqa: E402
    BotorchQLogNehviFiniteAcquisition,
)
from agent_evolve.ports.finite_acquisition import (  # noqa: E402
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
    FiniteAcquisitionRequest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_qlognehvi_selects_an_exact_unique_finite_batch() -> None:
    axes = (
        FiniteAcquisitionObjective("a", "min", 0.0, 2.0),
        FiniteAcquisitionObjective("b", "min", 0.0, 2.0),
    )
    observations = tuple(
        FiniteAcquisitionObservation(
            f"observed-{index}",
            _sha(f"observed-{index}"),
            (x, y),
            (("a", (x - 0.25) ** 2 + 0.1), ("b", (y - 0.75) ** 2 + 0.1)),
        )
        for index, (x, y) in enumerate(
            ((0.0, 0.0), (0.2, 0.8), (0.4, 0.4), (0.6, 0.2), (0.8, 0.6))
        )
    )
    observed_features = {value.features for value in observations}
    candidates = tuple(
        FiniteAcquisitionCandidate(
            f"candidate-{i}-{j}",
            _sha(f"candidate-{i}-{j}"),
            (i / 10.0, j / 10.0),
        )
        for i in range(1, 10)
        for j in range(1, 10)
        if (i / 10.0, j / 10.0) not in observed_features
    )
    request = FiniteAcquisitionRequest(
        campaign_scope_sha256=_sha("scope"),
        cutoff_index=1,
        batch_size=2,
        seed=11,
        objectives=axes,
        observations=observations,
        candidates=candidates,
    )

    decision = BotorchQLogNehviFiniteAcquisition(mc_samples=16).select(request)

    assert len(decision.selected) == 2
    assert len({value.candidate_id for value in decision.selected}) == 2
    assert {value.candidate_id for value in decision.selected}.issubset(
        {value.candidate_id for value in candidates}
    )
    assert decision.request_sha256 == request.request_sha256
