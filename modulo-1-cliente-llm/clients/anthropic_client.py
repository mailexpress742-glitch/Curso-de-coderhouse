"""Cliente de Anthropic (Messages API) sobre el SDK oficial asíncrono."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message
from pydantic import SecretStr

from clients.base import BaseLLMClient, error_from_status
from schemas import (
    ErrorKind,
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    Role,
    StreamChunk,
    Usage,
)

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(
        self,
        model: str,
        api_key: SecretStr | None = None,
        *,
        sdk_client: AsyncAnthropic | None = None,
        **retry_options: float,
    ) -> None:
        super().__init__(model, **retry_options)
        if sdk_client is None:
            # max_retries=0: los reintentos los maneja BaseLLMClient, así no se multiplican.
            sdk_client = AsyncAnthropic(
                api_key=api_key.get_secret_value() if api_key else None,
                max_retries=0,
            )
        self._client = sdk_client
        self._warned_temperature = False

    async def aclose(self) -> None:
        await self._client.close()

    def _request_kwargs(self, messages: Messages, params: GenerationParams) -> dict[str, Any]:
        # A diferencia de OpenAI, Anthropic recibe el prompt de sistema aparte,
        # no como un mensaje más de la lista.
        system = "\n\n".join(m.content for m in messages if m.role is Role.SYSTEM)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in messages
                if m.role is not Role.SYSTEM
            ],
            "timeout": params.timeout_s,
        }
        if system:
            kwargs["system"] = system
        if params.temperature is not None and not self._warned_temperature:
            # Los modelos Claude actuales rechazan `temperature` (y el SDK 1.x ya
            # no lo acepta), así que se ignora en vez de romper la llamada.
            logger.warning(
                "anthropic: 'temperature' no se envía; los modelos Claude actuales no lo aceptan"
            )
            self._warned_temperature = True
        return kwargs

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        message = await self._client.messages.create(**self._request_kwargs(messages, params))
        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content=_text_of(message),
            finish_reason=message.stop_reason,
            usage=_usage(message),
            error=_refusal_error(message),
        )

    async def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        async with self._client.messages.stream(**self._request_kwargs(messages, params)) as stream:
            # text_stream sólo emite el texto de la respuesta (no los bloques de razonamiento).
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)
            final = await stream.get_final_message()
        yield StreamChunk(
            done=True,
            finish_reason=final.stop_reason,
            usage=_usage(final),
            error=_refusal_error(final),
        )

    def _classify_error(self, exc: Exception) -> LLMError:
        # De la excepción más específica a la más general.
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return LLMError(
                kind=ErrorKind.AUTHENTICATION,
                message=f"API key inválida o sin permisos: {exc.message}",
                status_code=exc.status_code,
            )
        if isinstance(exc, anthropic.RateLimitError):
            return LLMError(
                kind=ErrorKind.RATE_LIMIT,
                message=f"límite de tasa alcanzado: {exc.message}",
                retryable=True,
                status_code=exc.status_code,
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(kind=ErrorKind.TIMEOUT, message="timeout de red", retryable=True)
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(
                kind=ErrorKind.CONNECTION, message=f"error de conexión: {exc}", retryable=True
            )
        if isinstance(exc, anthropic.APIStatusError):
            # Incluye 529 (OverloadedError), que error_from_status trata como reintentable.
            return error_from_status(exc.status_code, exc.message)
        return LLMError(kind=ErrorKind.UNKNOWN, message=f"{type(exc).__name__}: {exc}")


def _text_of(message: Message) -> str:
    # `content` es una lista de bloques (texto, razonamiento, ...): sólo interesa el texto.
    return "".join(block.text for block in message.content if block.type == "text")


def _usage(message: Message) -> Usage:
    return Usage(
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


def _refusal_error(message: Message) -> LLMError | None:
    if message.stop_reason != "refusal":
        return None
    category = message.stop_details.category if message.stop_details else None
    return LLMError(
        kind=ErrorKind.REFUSAL,
        message=f"el modelo declinó responder (categoría: {category or 'sin especificar'})",
    )
