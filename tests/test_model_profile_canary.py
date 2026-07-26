"""Provider-free checks for the generic model-profile canary boundary."""

from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    GPT_OSS_20B_GROQ_HIGH_SERIAL,
    MISTRAL_LARGE_3_MISTRAL,
)
from examples.development import run_model_profile_canary as canary


def test_canary_honors_profile_route_concurrency_cap() -> None:
    serial = canary._config(GPT_OSS_20B_GROQ_HIGH_SERIAL, seed=7)
    unconstrained = canary._config(MISTRAL_LARGE_3_MISTRAL, seed=7)

    assert serial.max_connections == 1
    assert unconstrained.max_connections == 2


def test_canary_selector_exposes_complete_200_option_wire_enum() -> None:
    selector, _ = canary._requests(
        run_id="provider_free_large_enum",
        profile=MISTRAL_LARGE_3_MISTRAL,
    )
    schema = selector.output_type.model_json_schema()
    option_schema = schema["$defs"]["_PortfolioMember"]["properties"][
        "option_id"
    ]

    assert option_schema["enum"] == list(canary._PORTFOLIO_OPTION_IDS)
    assert len(option_schema["enum"]) == 200
    assert len(selector.prompt.encode("utf-8")) < 2_048
