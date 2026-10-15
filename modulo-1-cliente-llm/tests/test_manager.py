import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr, ValidationError

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from config import Settings
from manager import AsyncLLMManager
from schemas import (
    ErrorKind,
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    StreamChunk,
)


class FakeClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(self, delay: float = 0, fail_mid_stream: bool = False) -> None:
        super().__init__("fake", retry_base_delay=0)
        self.delay = delay
        self.fail_mid_stream = fail_mid_stream
        self.calls = 0
        self.running = 0
        self.max_running = 0

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        self.calls += 1
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.running -= 1
        return ModelResponse(provider=self.provider, model=self.model, content="ok")

    async def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        yield StreamChunk(delta="hola ")
        if self.fail_mid_stream:
            raise ConnectionResetError("se cortó la conexión")
        yield StreamChunk(done=True, finish_reason="stop")

    def _classify_error(self, exc: Exception) -> LLMError:
        return LLMError(kind=ErrorKind.CONNECTION, message=str(exc), retryable=True)

    async def aclose(self) -> None:
        pass


def test_factory_picks_client_from_settings():
    openai_settings = Settings(provider="openai", openai_api_key=SecretStr("sk-x"))
    anthropic_settings = Settings(provider="anthropic", anthropic_api_key=SecretStr("sk-ant-x"))

    assert isinstance(AsyncLLMManager.from_settings(openai_settings).client, OpenAIClient)
    assert isinstance(AsyncLLMManager.from_settings(anthropic_settings).client, AnthropicClient)


def test_timeout_returns_error():
    client = FakeClient(delay=1)
    params = GenerationParams(timeout_s=0.05, max_retries=1)

    response = asyncio.run(client.generate([], params))

    assert response.error.kind == ErrorKind.TIMEOUT
    assert response.attempts == 2


def test_stream_is_not_retried_after_first_token():
    client = FakeClient(fail_mid_stream=True)

    async def collect():
        return [chunk async for chunk in client.stream([])]

    chunks = asyncio.run(collect())

    assert [c.delta for c in chunks] == ["hola ", ""]
    assert chunks[-1].error.kind == ErrorKind.CONNECTION
    assert client.calls == 1


def test_semaphore_limits_concurrency():
    client = FakeClient(delay=0.02)
    manager = AsyncLLMManager(client, max_concurrency=2)

    async def burst():
        return await asyncio.gather(*(manager.generate(f"pregunta {i}") for i in range(6)))

    responses = asyncio.run(burst())

    assert all(r.ok for r in responses)
    assert client.max_running == 2


def test_invalid_input_fails_before_calling_api():
    client = FakeClient()
    manager = AsyncLLMManager(client)

    with pytest.raises(ValidationError):
        asyncio.run(manager.generate("hola", temperature=5))
    assert client.calls == 0
