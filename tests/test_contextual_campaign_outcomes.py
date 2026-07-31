from __future__ import annotations

import asyncio
import hashlib
import math

from agent_evolve.application.contextual_campaign_outcomes import (
    observe_contextual_portfolio_outcomes,
)
from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.application.portfolio_evolution import PortfolioEvolution
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.policies.reward.contextual_marginal_utility import (
    ExactCoalitionShapleyContextualUtilityProjector,
    FixedReferenceContextualMarginalUtilityProjector,
)
from tests.test_portfolio_evolution import _build_wave


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Snapshot:
    def marginal_gain(self, point) -> float:
        return float(max(0.0, min(1.0, 1.0 - point["loss"] / 10.0)))

    def joint_gain(self, points) -> float:
        return max((self.marginal_gain(point) for point in points), default=0.0)


class _ArchiveUtility:
    def require_snapshot(self, value):
        assert type(value) is ArchiveUtilitySnapshot
        return _Snapshot()


def test_real_portfolio_result_projects_normalized_contextual_credit() -> None:
    async def scenario():
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "contextual-campaign-outcomes"
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave, defer_memory_credit=True)
        return wave, result

    wave, result = asyncio.run(scenario())
    batch = observe_contextual_portfolio_outcomes(
        campaign_scope_sha256=_sha("campaign"),
        wave_index=1,
        waves=(wave,),
        results=(result,),
        selected_source_ids=(("model", "engine", "model"),),
        marginal_utilities=((0.6, 0.0, 0.4),),
    )

    assert len(batch.observations) == 3
    assert sum(value.marginal_utility_share for value in batch.observations) == 1.0
    assert sum(value.positive_marginal_utility for value in batch.observations) == 2
    assert {value.source_id for value in batch.observations} == {"engine", "model"}
    assert {value.operator_id for value in batch.observations} == {
        "atomic",
        "composite",
    }
    option_by_candidate = {
        member.materialization.candidate_id.value: member.materialization.option_id
        for member in result.receipt.members
    }
    operator_by_option = {
        option_by_candidate[observation.candidate_id.value]: observation.operator_id
        for observation in batch.observations
    }
    assert operator_by_option["gamma.xy"] == "composite"
    assert all(value.final_front_persisted is None for value in batch.observations)
    assert all(
        value.useful_descendant_observed is None for value in batch.observations
    )
    assert batch.to_record()["policy"]["policy_id"] == (
        "normalized_fixed_reference_contextual_portfolio_outcomes"
    )

    receipt = freeze_json({"test": True})
    snapshot = ArchiveUtilitySnapshot(
        utility_id="test_contextual_utility",
        utility_version=1,
        definition_sha256=_sha("utility"),
        generation=1,
        benchmark_sha256=_sha("benchmark"),
        archive_sha256=_sha("archive"),
        snapshot_receipt=receipt,
    )
    utilities = FixedReferenceContextualMarginalUtilityProjector(
        _ArchiveUtility()
    ).project(snapshot=snapshot, results=(result,))
    assert len(utilities) == 1
    assert len(utilities[0]) == 3
    assert all(value >= 0.0 for value in utilities[0])

    shapley = ExactCoalitionShapleyContextualUtilityProjector(
        _ArchiveUtility()
    ).project(snapshot=snapshot, results=(result,))
    scored_points = tuple(
        outcome.candidate.objective_map
        for outcome in result.outcomes
        if outcome.candidate is not None
    )
    assert len(shapley) == 1
    assert len(shapley[0]) == 3
    assert all(value >= 0.0 for value in shapley[0])
    assert math.isclose(
        math.fsum(shapley[0]),
        _Snapshot().joint_gain(scored_points),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.fsum(shapley[0]) <= math.fsum(utilities[0])
