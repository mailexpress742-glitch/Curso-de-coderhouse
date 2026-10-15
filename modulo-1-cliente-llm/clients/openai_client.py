from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types import CompletionUsage

from clients import errors
from clients.base import BaseLLMClient
from schemas import (
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)

CONTENT_FILTER = "content_filter"


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        sdk_client: AsyncOpenAI | None = None,
        **retry_options: float,
    ) -> None:
        super().__init__(model, **retry_options)
        if sdk_client is None:
            # Sin reintentos en el SDK: ya los hace BaseLLMClient.
            sdk_client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self._sdk = sdk_client

    async def aclose(self) -> None:
        await self._sdk.close()

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        completion = await self._sdk.chat.completions.create(**self._request(messages, params))
        choice = completion.choices[0]
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=_usage(completion.usage),
            error=_refusal_error(choice.finish_reason, choice.message.refusal),
        )

    async def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._sdk.chat.completions.create(
            **self._request(messages, params),
            stream=True,
            stream_options={"include_usage": True},
        )
        finish_reason = None
        refusal = ""
        usage = None
        async with stream:
            async for chunk in stream:
                if chunk.usage:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    yield StreamChunk(delta=choice.delta.content)
                refusal += choice.delta.refusal or ""
                finish_reason = choice.finish_reason or finish_reason

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            usage=usage,
            error=_refusal_error(finish_reason, refusal),
        )

    def _classify_error(self, exc: Exception) -> LLMError:
        match exc:
            case openai.APITimeoutError():
                return errors.timeout()
            case openai.APIConnectionError():
                return errors.connection(exc)
            case openai.RateLimitError(code="insufficient_quota"):
                # También es un 429, pero sin saldo reintentar no sirve.
                error = errors.from_status(exc.status_code, exc.message)
                return error.model_copy(update={"retryable": False})
            case openai.APIStatusError():
                return errors.from_status(exc.status_code, exc.message)
            case _:
                return errors.unknown(exc)

    def _request(self, messages: Messages, params: GenerationParams) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            # Los modelos de razonamiento no aceptan max_tokens.
            "max_completion_tokens": params.max_tokens,
            "timeout": params.timeout_s,
        }
        if params.temperature is not None:
            request["temperature"] = params.temperature
        return request


def _usage(usage: CompletionUsage | None) -> Usage | None:
    if usage is None:
        return None
    return Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)


def _refusal_error(finish_reason: str | None, refusal: str | None) -> LLMError | None:
    if refusal:
        return errors.refusal(refusal)
    if finish_reason == CONTENT_FILTER:
        return errors.refusal("filtro de contenido")
    return None
