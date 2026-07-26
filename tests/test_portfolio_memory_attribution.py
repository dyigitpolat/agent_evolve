from __future__ import annotations

import asyncio
from dataclasses import replace

from agent_evolve.application.portfolio_evolution import PortfolioEvolution
from agent_evolve.application.portfolio_memory_attribution import (
    PortfolioMemoryAttributionAudit,
    audit_portfolio_memory_attribution,
)
from tests.test_portfolio_evolution import (
    _build_wave,
    _contract,
    _rebind_credit_plan,
)


def test_generation_audit_separates_member_contribution_from_wave_credit() -> None:
    async def scenario():
        values = await _build_wave("memory_attribution_two_lane")
        ids, _problem, _generator, memory, engine, selector, first = values
        second_parent = await engine.register_seed(
            {"x": 10, "y": 10},
            label="memory_attribution_second_parent",
        )
        first_credit = first.memory_credit
        assert first_credit is not None
        second_request = replace(
            first.selection_request,
            call_id=ids.new_llm_call_id(),
            finite_variation_contract=_contract(second_parent.configuration),
        )
        second = replace(
            first,
            selection_request=second_request,
            parent=second_parent,
            label_prefix="memory_attribution_second",
            memory_credit=_rebind_credit_plan(
                first_credit,
                credit_unit_id=ids.new_operator_invocation_id(),
            ),
        )
        service = PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        )
        pending = tuple(
            await asyncio.gather(
                service.run(first, defer_memory_credit=True),
                service.run(second, defer_memory_credit=True),
            )
        )
        committed, _batch = service.commit_pending_memory_credit_batch(pending)
        return (first, second), committed

    waves, results = asyncio.run(scenario())
    audit = audit_portfolio_memory_attribution(waves=waves, results=results)

    assert type(audit) is PortfolioMemoryAttributionAudit
    assert len(audit.candidate_contributions) == 6
    assert len(audit.card_performance) == 6
    record = audit.to_record()
    assert record["causal_card_effect_identified"] is False
    assert record["causal_action_effect_identified"] is False
    assert record["online_score_update_allowed"] is False
    # Both deterministic lanes select the same three option IDs, so every
    # explicit card/action row has a cross-lane realization.
    assert record["cross_lane_action_spillover_count"] == 6
    assert all(
        value.cross_lane_action_spillover for value in audit.card_performance
    )

    result_by_request = {value.receipt.request_sha256: value for value in results}
    wave_by_request = {
        value.selection_request.request_sha256: value for value in waves
    }
    for contribution in audit.candidate_contributions:
        result = result_by_request[contribution.request_sha256]
        wave = wave_by_request[contribution.request_sha256]
        credit = wave.memory_credit
        assert credit is not None
        index = contribution.rank - 1
        full = credit.aggregation.aggregate(result.outcomes)
        leave_one_out = credit.aggregation.aggregate(
            result.outcomes[:index] + result.outcomes[index + 1 :]
        )
        assert contribution.joint_wave_reward == full
        assert contribution.leave_one_out_wave_reward == leave_one_out
        assert contribution.leave_one_out_contribution == full - leave_one_out


def test_no_memory_wave_has_no_memory_attribution_audit() -> None:
    async def scenario():
        values = await _build_wave("memory_attribution_absent")
        ids, _problem, _generator, memory, engine, selector, wave = values
        wave = replace(wave, memory_credit=None)
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave)
        return wave, result

    wave, result = asyncio.run(scenario())
    assert (
        audit_portfolio_memory_attribution(
            waves=(wave,),
            results=(result,),
        )
        is None
    )
