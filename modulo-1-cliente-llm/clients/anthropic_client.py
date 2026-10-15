import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message

from clients import errors
from clients.base import BaseLLMClient
from schemas import (
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
        api_key: str | None = None,
        *,
        sdk_client: AsyncAnthropic | None = None,
        **retry_options: float,
    ) -> None:
        super().__init__(model, **retry_options)
        if sdk_client is None:
            # Sin reintentos en el SDK: ya los hace BaseLLMClient.
            sdk_client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self._sdk = sdk_client

    async def aclose(self) -> None:
        await self._sdk.close()

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        message = await self._sdk.messages.create(**self._request(messages, params))
        return _to_response(message)

    async def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        async with self._sdk.messages.stream(**self._request(messages, params)) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)
            message = await stream.get_final_message()

        response = _to_response(message)
        yield StreamChunk(
            done=True,
            finish_reason=response.finish_reason,
            usage=response.usage,
            error=response.error,
        )

    def _classify_error(self, exc: Exception) -> LLMError:
        match exc:
            case anthropic.APITimeoutError():
                return errors.timeout()
            case anthropic.APIConnectionError():
                return errors.connection(exc)
            case anthropic.APIStatusError():
                return errors.from_status(exc.status_code, exc.message)
            case _:
                return errors.unknown(exc)

    def _request(self, messages: Messages, params: GenerationParams) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != Role.SYSTEM
            ],
            "timeout": params.timeout_s,
        }
        # Anthropic recibe el prompt de sistema aparte, no dentro de messages.
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        if system:
            request["system"] = system
        if params.temperature is not None:
            logger.warning("anthropic no acepta temperature en los modelos actuales; se ignora")
        return request


def _to_response(message: Message) -> ModelResponse:
    # content es una lista de bloques; los de razonamiento no se muestran.
    text = "".join(block.text for block in message.content if block.type == "text")
    return ModelResponse(
        provider=Provider.ANTHROPIC,
        model=message.model,
        content=text,
        finish_reason=message.stop_reason,
        usage=Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        ),
        error=_refusal_error(message),
    )


def _refusal_error(message: Message) -> LLMError | None:
    if message.stop_reason != "refusal":
        return None
    category = message.stop_details.category if message.stop_details else None
    return errors.refusal(f"categoría: {category or 'sin especificar'}")
