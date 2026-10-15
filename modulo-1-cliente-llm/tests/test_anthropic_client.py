import asyncio

from anthropic import AsyncAnthropic

from clients import AnthropicClient
from fakes import FakeTransport, anthropic_error, anthropic_message, anthropic_stream
from schemas import ChatMessage, ErrorKind, GenerationParams, Role

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Sos breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def make_client(transport: FakeTransport) -> AnthropicClient:
    sdk = AsyncAnthropic(api_key="sk-ant-test", max_retries=0, http_client=transport.http_client())
    return AnthropicClient("claude-opus-5-5", sdk_client=sdk, retry_base_delay=0)


def generate(transport: FakeTransport, params: GenerationParams | None = None):
    return asyncio.run(make_client(transport).generate(MESSAGES, params))


def stream(transport: FakeTransport):
    async def collect():
        return [chunk async for chunk in make_client(transport).stream(MESSAGES)]

    return asyncio.run(collect())


def test_generate_returns_only_text_blocks():
    transport = FakeTransport(anthropic_message())

    response = generate(transport)

    assert response.ok
    assert response.content == "Es una medida del desorden."
    assert response.finish_reason == "end_turn"


def test_system_prompt_is_sent_separately():
    transport = FakeTransport(anthropic_message())

    generate(transport)

    body = transport.sent_body()
    assert body["system"] == "Sos breve."
    assert body["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]


def test_temperature_is_ignored():
    transport = FakeTransport(anthropic_message())

    response = generate(transport, GenerationParams(temperature=0.7))

    assert response.ok
    assert "temperature" not in transport.sent_body()


def test_stream():
    transport = FakeTransport(anthropic_stream("La ", "entropía ", "mide..."))

    chunks = stream(transport)

    assert "".join(c.delta for c in chunks) == "La entropía mide..."
    last = chunks[-1]
    assert last.done
    assert last.error is None
    assert last.finish_reason == "end_turn"
    assert last.usage.output_tokens == 4


def test_retries_when_overloaded():
    transport = FakeTransport(anthropic_error(529, "overloaded_error"), anthropic_message())

    response = generate(transport)

    assert response.ok
    assert response.attempts == 2


def test_respects_retry_after_header():
    rate_limited = anthropic_error(429, "rate_limit_error")
    rate_limited.headers["retry-after"] = "0"
    transport = FakeTransport(rate_limited, anthropic_message())

    response = generate(transport)

    assert response.ok
    assert response.attempts == 2


def test_invalid_request_is_not_retried():
    transport = FakeTransport(anthropic_error(404, "not_found_error"))

    response = generate(transport)

    assert response.error.kind == ErrorKind.INVALID_REQUEST
    assert len(transport.requests) == 1


def test_refusal_returns_error():
    stop_details = {"type": "refusal", "category": "cyber", "explanation": None}
    transport = FakeTransport(anthropic_message("refusal", stop_details=stop_details))

    response = generate(transport)

    assert response.error.kind == ErrorKind.REFUSAL
    assert "cyber" in response.error.message
