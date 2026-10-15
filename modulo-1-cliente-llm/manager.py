import asyncio
from collections.abc import AsyncIterator
from typing import Any, Self

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from config import Settings
from schemas import (
    ChatMessage,
    ChatRequest,
    GenerationParams,
    Messages,
    ModelResponse,
    Provider,
    Role,
    StreamChunk,
)

CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}


def create_client(settings: Settings) -> BaseLLMClient:
    client_class = CLIENTS[settings.provider]
    return client_class(model=settings.model, api_key=settings.api_key.get_secret_value())


class AsyncLLMManager:
    """Punto de entrada para el resto de la aplicación.

    El proveedor se elige por configuración (LLM_PROVIDER), así que cambiar de
    OpenAI a Anthropic no requiere tocar el código que usa el manager.
    """

    def __init__(self, client: BaseLLMClient, *, max_concurrency: int = 5) -> None:
        self.client = client
        # Limita las llamadas simultáneas para no gatillar errores 429.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(create_client(settings), max_concurrency=settings.max_concurrency)

    @classmethod
    def from_env(cls, provider: str | None = None) -> Self:
        return cls.from_settings(Settings.from_env(provider))

    @property
    def provider(self) -> Provider:
        return self.client.provider

    @property
    def model(self) -> str:
        return self.client.model

    async def generate(
        self,
        messages: Messages | str,
        params: GenerationParams | None = None,
        **overrides: Any,
    ) -> ModelResponse:
        request = _build_request(messages, params, overrides)
        async with self._semaphore:
            return await self.client.generate(request.messages, request.params)

    async def stream(
        self,
        messages: Messages | str,
        params: GenerationParams | None = None,
        **overrides: Any,
    ) -> AsyncIterator[StreamChunk]:
        request = _build_request(messages, params, overrides)
        async with self._semaphore:
            async for chunk in self.client.stream(request.messages, request.params):
                yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _build_request(
    messages: Messages | str,
    params: GenerationParams | None,
    overrides: dict[str, Any],
) -> ChatRequest:
    """Valida la entrada. Si algo está mal lanza ValidationError antes de llamar a la API."""
    if isinstance(messages, str):
        messages = [ChatMessage(role=Role.USER, content=messages)]
    base = params.model_dump(exclude_unset=True) if params else {}
    return ChatRequest(
        messages=list(messages),
        params=GenerationParams.model_validate(base | overrides),
    )
