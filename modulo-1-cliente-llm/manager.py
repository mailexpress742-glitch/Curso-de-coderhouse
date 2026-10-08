"""AsyncLLMManager: punto de entrada único para hablar con cualquier proveedor.

El resto de la aplicación sólo conoce esta clase. Qué proveedor se usa se decide
por configuración (LLM_PROVIDER), así que pasar de OpenAI a Anthropic no
requiere cambiar ninguna línea de lógica de negocio.
"""

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

# Factory: qué clase concreta corresponde a cada proveedor.
CLIENTS: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}


def create_client(settings: Settings) -> BaseLLMClient:
    client_cls = CLIENTS[settings.provider]
    return client_cls(
        model=settings.model_for(settings.provider),
        api_key=settings.api_key_for(settings.provider),
    )


class AsyncLLMManager:
    def __init__(self, client: BaseLLMClient, *, max_concurrency: int = 5) -> None:
        self.client = client
        # Tope de llamadas simultáneas: protege la cuota (429) si muchas tareas
        # usan el mismo manager a la vez.
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(create_client(settings), max_concurrency=settings.max_concurrency)

    @classmethod
    def from_env(cls, provider: Provider | str | None = None) -> Self:
        """Lee la configuración del entorno. Lanza ValidationError si está incompleta."""
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
        """Respuesta completa.

        Las entradas inválidas (p. ej. temperature=3) lanzan ValidationError antes
        de llamar a la API: son errores del que llama. Los fallos de red, cuota o
        autenticación no lanzan: vuelven en `ModelResponse.error`.
        """
        request = _build_request(messages, params, overrides)
        async with self._semaphore:
            return await self.client.generate(request.messages, request.params)

    async def stream(
        self,
        messages: Messages | str,
        params: GenerationParams | None = None,
        **overrides: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono de chunks de texto, a medida que llegan de la API."""
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
    messages: Messages | str, params: GenerationParams | None, overrides: dict[str, Any]
) -> ChatRequest:
    if isinstance(messages, str):
        messages = [ChatMessage(role=Role.USER, content=messages)]
    base = params.model_dump(exclude_unset=True) if params else {}
    return ChatRequest(
        messages=list(messages),
        params=GenerationParams.model_validate(base | overrides),
    )
