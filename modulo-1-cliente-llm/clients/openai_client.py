"""Cliente de OpenAI (Chat Completions) sobre el SDK oficial asíncrono."""

from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from pydantic import SecretStr

from clients.base import BaseLLMClient, error_from_status
from schemas import (
    ErrorKind,
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    StreamChunk,
    Usage,
)

# OpenAI corta la respuesta con este finish_reason cuando la filtra por contenido.
_CONTENT_FILTER = "content_filter"


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(
        self,
        model: str,
        api_key: SecretStr | None = None,
        *,
        sdk_client: AsyncOpenAI | None = None,
        **retry_options: float,
    ) -> None:
        super().__init__(model, **retry_options)
        if sdk_client is None:
            # max_retries=0: los reintentos los maneja BaseLLMClient, así no se multiplican.
            sdk_client = AsyncOpenAI(
                api_key=api_key.get_secret_value() if api_key else None,
                max_retries=0,
            )
        self._client = sdk_client

    async def aclose(self) -> None:
        await self._client.close()

    def _request_kwargs(self, messages: Messages, params: GenerationParams) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            # max_completion_tokens reemplaza a max_tokens y es el que aceptan los
            # modelos de razonamiento.
            "max_completion_tokens": params.max_tokens,
            "timeout": params.timeout_s,
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        return kwargs

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        completion = await self._client.chat.completions.create(
            **self._request_kwargs(messages, params)
        )
        choice = completion.choices[0]
        error = None
        if choice.finish_reason == _CONTENT_FILTER or choice.message.refusal:
            error = _refusal_error(choice.message.refusal)
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=_usage(completion.usage),
            error=error,
        )

    async def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            **self._request_kwargs(messages, params),
            stream=True,
            stream_options={"include_usage": True},  # el último chunk trae el uso de tokens
        )
        finish_reason: str | None = None
        refusal = ""
        usage: Usage | None = None
        async with stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    yield StreamChunk(delta=choice.delta.content)
                if choice.delta.refusal:
                    refusal += choice.delta.refusal
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        error = None
        if finish_reason == _CONTENT_FILTER or refusal:
            error = _refusal_error(refusal)
        yield StreamChunk(done=True, finish_reason=finish_reason, usage=usage, error=error)

    def _classify_error(self, exc: Exception) -> LLMError:
        # De la excepción más específica a la más general.
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return LLMError(
                kind=ErrorKind.AUTHENTICATION,
                message=f"API key inválida o sin permisos: {exc.message}",
                status_code=exc.status_code,
            )
        if isinstance(exc, openai.RateLimitError):
            # Sin saldo también es un 429, pero reintentar no lo arregla.
            out_of_quota = exc.code == "insufficient_quota"
            return LLMError(
                kind=ErrorKind.RATE_LIMIT,
                message=f"límite de tasa o cuota alcanzado: {exc.message}",
                retryable=not out_of_quota,
                status_code=exc.status_code,
            )
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(kind=ErrorKind.TIMEOUT, message="timeout de red", retryable=True)
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(
                kind=ErrorKind.CONNECTION, message=f"error de conexión: {exc}", retryable=True
            )
        if isinstance(exc, openai.APIStatusError):
            return error_from_status(exc.status_code, exc.message)
        return LLMError(kind=ErrorKind.UNKNOWN, message=f"{type(exc).__name__}: {exc}")


def _usage(usage: CompletionUsage | None) -> Usage | None:
    if usage is None:
        return None
    return Usage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)


def _refusal_error(refusal: str | None) -> LLMError:
    detail = refusal or "la respuesta fue filtrada por contenido"
    return LLMError(kind=ErrorKind.REFUSAL, message=f"el modelo declinó responder: {detail}")
