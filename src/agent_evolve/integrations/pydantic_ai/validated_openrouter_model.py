"""Pinned OpenRouter stream-item compatibility boundary.

The OpenAI SDK's streamed decoder can yield an arbitrary JSON primitive when
an OpenAI-compatible endpoint sends a syntactically valid SSE ``data`` field
whose JSON value is not a chat-completion object.  Pydantic-AI 1.107.1 assumes
that every decoded value is a ``ChatCompletionChunk`` before validating its
runtime type.  This adapter rejects that narrow malformed wire topology
without broadening retry semantics to unrelated ``AttributeError`` or
validation failures.

The underlying SDK stream remains owned and closed by Pydantic-AI's
``request_stream`` context manager.  The wrapper is single-pass and validates
each decoded SDK item immediately before exposing it to Pydantic-AI.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import cast

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import (
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)
from pydantic_ai.models.openrouter import OpenRouterModel


class InvalidOpenRouterStreamItemError(RuntimeError):
    """A payload-free signal for a non-chat-completion decoded SSE item."""

    __slots__ = ()

    def __init__(self) -> None:
        # The decoded item, its rendering, and its provider payload must never
        # become exception state or scientific telemetry.
        super().__init__("OpenRouter stream yielded an invalid item")


class _ValidatedChatCompletionChunkStream(AsyncIterator[ChatCompletionChunk]):
    """Single-pass view that validates every decoded SDK stream item."""

    __slots__ = (
        "_closed",
        "_source",
        "_source_iterator",
    )

    def __init__(self, source: AsyncStream[ChatCompletionChunk]) -> None:
        # Keep the original AsyncStream: Pydantic-AI's streamed response closes
        # ``PeekableAsyncStream.source`` through its async ``close`` method.
        self._source = source
        self._source_iterator = aiter(cast(AsyncIterable[object], source))
        self._closed = False

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        item = await anext(self._source_iterator)
        if type(item) is not ChatCompletionChunk:
            raise InvalidOpenRouterStreamItemError()
        return cast(ChatCompletionChunk, item)

    async def close(self) -> None:
        """Delegate Pydantic-AI's close contract to the owned SDK stream."""

        if self._closed:
            return
        self._closed = True
        await self._source.close()

    async def aclose(self) -> None:
        """Support generic async-iterator cleanup without changing ownership."""

        await self.close()


class ValidatedOpenRouterModel(OpenRouterModel):
    """OpenRouterModel with a narrow decoded stream-item boundary.

    The override is intentionally isolated because ``_process_streamed_response``
    is a private Pydantic-AI compatibility seam.  ``pydantic-ai==1.107.1`` is
    pinned by this project; conformance tests must fail loudly if that seam
    changes on a future dependency upgrade.
    """

    async def _process_streamed_response(
        self,
        response: AsyncStream[ChatCompletionChunk],
        model_request_parameters: ModelRequestParameters,
        model_settings: OpenAIChatModelSettings | None = None,
    ) -> OpenAIStreamedResponse:
        validated = _ValidatedChatCompletionChunkStream(response)
        return await super()._process_streamed_response(
            cast(AsyncStream[ChatCompletionChunk], validated),
            model_request_parameters,
            model_settings,
        )


__all__ = [
    "InvalidOpenRouterStreamItemError",
    "ValidatedOpenRouterModel",
]
