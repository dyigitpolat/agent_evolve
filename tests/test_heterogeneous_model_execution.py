from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.integrations.pydantic_ai.heterogeneous_model_execution import (
    HeterogeneousModelExecutionProfile,
    HeterogeneousRunnerConfig,
    ModelLaneBinding,
    compose_operation_suffix_dispatching_runner,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    openrouter_model_execution_profile,
)


@dataclass(frozen=True, slots=True)
class _Request:
    call_id: LLMCallId
    operation: str


class _Runner:
    def __init__(self, lane_id: str, seed: int) -> None:
        self.lane_id = lane_id
        self.seed = seed
        self.closed = False

    async def __call__(self, request: _Request) -> tuple[str, int, str]:
        return self.lane_id, self.seed, request.call_id.value

    async def aclose(self) -> None:
        self.closed = True


def _profile() -> HeterogeneousModelExecutionProfile:
    return HeterogeneousModelExecutionProfile(
        profile_id="qwen_deepseek_test",
        lanes=(
            ModelLaneBinding(
                lane_id="deepseek",
                profile=openrouter_model_execution_profile("deepseek_json"),
            ),
            ModelLaneBinding(
                lane_id="qwen",
                profile=openrouter_model_execution_profile("qwen_rate_safe"),
            ),
        ),
    )


def test_routes_primary_and_repair_to_independent_lanes() -> None:
    created: dict[str, _Runner] = {}

    def factory(binding: ModelLaneBinding, seed: int) -> _Runner:
        runner = _Runner(binding.lane_id, seed)
        created[binding.lane_id] = runner
        return runner

    async def exercise() -> None:
        runner = compose_operation_suffix_dispatching_runner(
            profile=_profile(),
            config=HeterogeneousRunnerConfig(seed=11),
            lane_runner_factory=factory,
        )
        primary = await runner(
            _Request(
                call_id=LLMCallId("call_qwen"),
                operation="propose_residual_plans_qwen",
            )
        )
        repair = await runner(
            _Request(
                call_id=LLMCallId("call_qwen_reground"),
                operation="postcompile_semantic_regrounding",
            )
        )
        deepseek = await runner(
            _Request(
                call_id=LLMCallId("call_deepseek"),
                operation="propose_residual_plans_deepseek",
            )
        )
        assert primary == ("qwen", 12, "call_qwen")
        assert repair == ("qwen", 12, "call_qwen_reground")
        assert deepseek == ("deepseek", 11, "call_deepseek")
        await runner.aclose()
        assert all(value.closed for value in created.values())

    asyncio.run(exercise())


def test_rejects_unbound_operation_and_orphan_repair() -> None:
    async def exercise() -> None:
        runner = compose_operation_suffix_dispatching_runner(
            profile=_profile(),
            config=HeterogeneousRunnerConfig(seed=0),
            lane_runner_factory=lambda binding, seed: _Runner(
                binding.lane_id,
                seed,
            ),
        )
        with pytest.raises(
            ValueError,
            match="no authenticated model-lane route",
        ):
            await runner(
                _Request(
                    call_id=LLMCallId("call_orphan_reground"),
                    operation="postcompile_semantic_regrounding",
                )
            )
        with pytest.raises(
            ValueError,
            match="no authenticated model-lane route",
        ):
            await runner(
                _Request(
                    call_id=LLMCallId("call_ambiguous"),
                    operation="propose_residual_plans",
                )
            )

    asyncio.run(exercise())


def test_profile_record_is_canonical_and_workload_free() -> None:
    profile = _profile()
    record = profile.to_record()
    assert profile.lane_ids == ("deepseek", "qwen")
    assert record["workload_specific_fields"] == []
    assert record["independent_provider_queues"] is True
    assert record["profile_sha256"] == profile.profile_sha256


def test_profile_rejects_duplicate_model_identity() -> None:
    qwen = openrouter_model_execution_profile("qwen_rate_safe")
    with pytest.raises(ValueError, match="requested model"):
        HeterogeneousModelExecutionProfile(
            profile_id="duplicate_model_test",
            lanes=(
                ModelLaneBinding(lane_id="a", profile=qwen),
                ModelLaneBinding(lane_id="b", profile=qwen),
            ),
        )
