"""Model-route-owned JSON-schema compatibility transformations.

Provider-native structured output APIs do not all implement the same JSON
Schema vocabulary.  These transformations affect only the schema sent on the
wire; AgentEvolve still validates the returned object against the original
Pydantic type, so semantic and validation constraints remain authoritative.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class OpenRouterJsonSchemaDialect(str, Enum):
    """Closed wire-schema dialect selected by a model execution profile."""

    PROVIDER_DEFAULT = "provider_default"
    ALIBABA_NATIVE_V1 = "alibaba_native_v1"
    OPENAI_STRICT_BOUNDED_TEXT_V1 = "openai_strict_bounded_text_v1"


def json_schema_transformer_for_dialect(
    base_transformer: type[Any],
    dialect: OpenRouterJsonSchemaDialect,
) -> type[Any]:
    """Compose a provider dialect over Pydantic-AI's model transformer.

    Alibaba's native JSON-schema endpoint currently rejects the standard
    ``uniqueItems`` array keyword.  Dropping it is safe here because the
    request's original Pydantic output type performs the uniqueness check
    after decoding.  The inherited transformer continues to own every other
    model-profile transformation, including definition inlining.
    """

    if not isinstance(base_transformer, type):
        raise TypeError("base_transformer must be a class")
    if type(dialect) is not OpenRouterJsonSchemaDialect:
        raise TypeError("dialect must be an exact OpenRouterJsonSchemaDialect")
    if dialect is OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT:
        return base_transformer

    if dialect is OpenRouterJsonSchemaDialect.OPENAI_STRICT_BOUNDED_TEXT_V1:

        class OpenAIStrictBoundedTextJsonSchemaTransformer(base_transformer):
            """Encode unsupported string-length bounds in an allowed pattern.

            Pydantic-AI's OpenAI strict transformer correctly removes
            ``minLength`` and ``maxLength`` because OpenAI's strict subset does
            not accept those keywords.  A bounded anchored pattern retains the
            same generation constraint for unconstrained free-text fields.
            Fields that already own a semantic pattern are left unchanged;
            local Pydantic validation remains authoritative in every case.
            """

            def transform(self, schema: Any) -> Any:
                if (
                    isinstance(schema, dict)
                    and self.strict is True
                    and schema.get("type") == "string"
                    and "pattern" not in schema
                ):
                    minimum = schema.get("minLength", 0)
                    maximum = schema.get("maxLength")
                    if (
                        type(minimum) is int
                        and minimum >= 0
                        and type(maximum) is int
                        and minimum <= maximum <= 65_536
                    ):
                        schema = dict(schema)
                        # Newlines are not useful in AgentEvolve's bounded
                        # rationale/insight atoms and avoiding DOTALL-specific
                        # constructs keeps the provider grammar portable.
                        schema["pattern"] = f"^.{{{minimum},{maximum}}}$"
                return super().transform(schema)

        OpenAIStrictBoundedTextJsonSchemaTransformer.__name__ = (
            "OpenAIStrictBoundedTextJsonSchemaTransformer"
        )
        OpenAIStrictBoundedTextJsonSchemaTransformer.__qualname__ = (
            "OpenAIStrictBoundedTextJsonSchemaTransformer"
        )
        OpenAIStrictBoundedTextJsonSchemaTransformer.__module__ = __name__
        return OpenAIStrictBoundedTextJsonSchemaTransformer

    class AlibabaNativeJsonSchemaTransformer(base_transformer):
        def transform(self, schema: Any) -> Any:
            transformed = super().transform(schema)
            if isinstance(transformed, dict) and "uniqueItems" in transformed:
                transformed = dict(transformed)
                transformed.pop("uniqueItems")
            return transformed

    AlibabaNativeJsonSchemaTransformer.__name__ = (
        "AlibabaNativeJsonSchemaTransformer"
    )
    AlibabaNativeJsonSchemaTransformer.__qualname__ = (
        "AlibabaNativeJsonSchemaTransformer"
    )
    AlibabaNativeJsonSchemaTransformer.__module__ = __name__
    return AlibabaNativeJsonSchemaTransformer


__all__ = [
    "OpenRouterJsonSchemaDialect",
    "json_schema_transformer_for_dialect",
]
