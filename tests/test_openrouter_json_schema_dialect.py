"""Provider-dialect regressions for workload-neutral structured outputs."""

from __future__ import annotations

from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from agent_evolve.integrations.pydantic_ai.json_schema_dialect import (
    OpenRouterJsonSchemaDialect,
    json_schema_transformer_for_dialect,
)


def test_strict_bounded_text_dialect_preserves_generation_length_as_pattern() -> None:
    transformer = json_schema_transformer_for_dialect(
        OpenAIJsonSchemaTransformer,
        OpenRouterJsonSchemaDialect.OPENAI_STRICT_BOUNDED_TEXT_V1,
    )
    schema = {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
            },
            "identifier": {
                "type": "string",
                "pattern": "^option_[0-9]+$",
                "maxLength": 32,
            },
        },
        "required": ["rationale", "identifier"],
    }

    wire = transformer(schema, strict=True).walk()

    rationale = wire["properties"]["rationale"]
    assert rationale["pattern"] == "^.{1,512}$"
    assert "minLength" not in rationale
    assert "maxLength" not in rationale
    assert wire["properties"]["identifier"]["pattern"] == "^option_[0-9]+$"
    assert wire["required"] == ["rationale", "identifier"]


def test_bounded_text_dialect_does_not_change_non_strict_wire_schema() -> None:
    transformer = json_schema_transformer_for_dialect(
        OpenAIJsonSchemaTransformer,
        OpenRouterJsonSchemaDialect.OPENAI_STRICT_BOUNDED_TEXT_V1,
    )
    wire = transformer(
        {"type": "string", "minLength": 2, "maxLength": 9},
        strict=False,
    ).walk()

    assert wire == {"type": "string", "minLength": 2, "maxLength": 9}
