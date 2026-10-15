import asyncio

import httpx2
from openai import AsyncOpenAI

from clients import OpenAIClient
from fakes import FakeTransport, openai_completion, openai_error, openai_stream
from schemas import ChatMessage, ErrorKind, GenerationParams, Role

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Sos breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def make_client(transport: FakeTransport) -> OpenAIClient:
    sdk = AsyncOpenAI(api_key="sk-test", max_retries=0, http_client=transport.http_client())
    return OpenAIClient("gpt-5-mini", sdk_client=sdk, retry_base_delay=0)


def generate(transport: FakeTransport, params: GenerationParams | None = None):
    return asyncio.run(make_client(transport).generate(MESSAGES, params))


def stream(transport: FakeTransport):
    async def collect():
        return [chunk async for chunk in make_client(transport).stream(MESSAGES)]

    return asyncio.run(collect())


def test_generate():
    transport = FakeTransport(openai_completion("Es una medida del desorden."))

    response = generate(transport)

    assert response.ok
    assert response.content == "Es una medida del desorden."
    assert response.usage.output_tokens == 8
    body = transport.sent_body()
    assert body["messages"][0] == {"role": "system", "content": "Sos breve."}
    assert body["max_completion_tokens"] == 16_000
    assert "temperature" not in body


def test_temperature_is_sent_only_when_set():
    transport = FakeTransport(openai_completion())
    generate(transport, GenerationParams(temperature=0.2))
    assert transport.sent_body()["temperature"] == 0.2


def test_stream():
    transport = FakeTransport(openai_stream("La ", "entropía ", "mide..."))

    chunks = stream(transport)

    assert "".join(c.delta for c in chunks) == "La entropía mide..."
    last = chunks[-1]
    assert last.done
    assert last.error is None
    assert last.finish_reason == "stop"
    assert last.usage.output_tokens == 3


def test_retries_on_rate_limit():
    transport = FakeTransport(openai_error(429, "rate_limit_exceeded"), openai_completion())

    response = generate(transport)

    assert response.ok
    assert response.attempts == 2


def test_does_not_retry_without_quota():
    transport = FakeTransport(openai_error(429, "insufficient_quota"))

    response = generate(transport)

    assert response.error.kind == ErrorKind.RATE_LIMIT
    assert not response.error.retryable
    assert len(transport.requests) == 1


def test_invalid_api_key_returns_error():
    transport = FakeTransport(openai_error(401, "invalid_api_key"))

    response = generate(transport)

    assert response.error.kind == ErrorKind.AUTHENTICATION
    assert response.error.status_code == 401
    assert len(transport.requests) == 1


def test_connection_error_after_all_retries():
    transport = FakeTransport(*[httpx2.ConnectError("sin red") for _ in range(3)])

    response = generate(transport, GenerationParams(max_retries=2))

    assert response.error.kind == ErrorKind.CONNECTION
    assert response.attempts == 3


def test_stream_retries_before_first_token():
    transport = FakeTransport(openai_error(500, "server_error"), openai_stream("hola"))

    chunks = stream(transport)

    assert "".join(c.delta for c in chunks) == "hola"
    assert chunks[-1].error is None


def test_stream_returns_error_chunk():
    transport = FakeTransport(openai_error(401, "invalid_api_key"))

    chunks = stream(transport)

    assert len(chunks) == 1
    assert chunks[0].done
    assert chunks[0].error.kind == ErrorKind.AUTHENTICATION
