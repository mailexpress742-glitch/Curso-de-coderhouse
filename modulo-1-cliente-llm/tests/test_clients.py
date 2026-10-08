"""Tests sin API keys ni red.

Los clientes reales de OpenAI y Anthropic se prueban con los SDKs oficiales
apuntando a un transporte HTTP simulado (httpx2.MockTransport): así se verifica
el request que se arma, el parseo de la respuesta y del streaming SSE, y el
manejo de errores, sin gastar tokens.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import SecretStr, ValidationError

from clients import AnthropicClient, BaseLLMClient, OpenAIClient
from config import Settings
from manager import AsyncLLMManager
from schemas import (
    ChatMessage,
    ChatRequest,
    ErrorKind,
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    Role,
    StreamChunk,
)

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Sos breve."),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]

Handler = Callable[[httpx2.Request], httpx2.Response]


def run(coro):
    return asyncio.run(coro)


async def collect(stream: AsyncIterator[StreamChunk]) -> list[StreamChunk]:
    return [chunk async for chunk in stream]


def sse(events: list[tuple[str | None, object]]) -> bytes:
    lines = []
    for event, data in events:
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


class Recorder:
    """Devuelve respuestas en orden y guarda los requests recibidos."""

    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def openai_client(handler: Handler) -> OpenAIClient:
    sdk = AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIClient("gpt-5-mini", sdk_client=sdk, retry_base_delay=0)


def anthropic_client(handler: Handler) -> AnthropicClient:
    sdk = AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicClient("claude-opus-5-5", sdk_client=sdk, retry_base_delay=0)


# ----------------------------------------------------------------------
# Respuestas simuladas
# ----------------------------------------------------------------------

def openai_completion(content: str = "Es una medida del desorden.", finish: str = "stop") -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5-mini",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content, "refusal": None}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    })


def openai_stream(*deltas: str) -> httpx2.Response:
    def chunk(delta: dict, finish: str | None = None) -> dict:
        return {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-5-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    events: list[tuple[str | None, object]] = [(None, chunk({"role": "assistant", "content": ""}))]
    events += [(None, chunk({"content": d})) for d in deltas]
    events.append((None, chunk({}, "stop")))
    events.append((None, {"id": "c1", "object": "chat.completion.chunk", "created": 0,
                          "model": "gpt-5-mini", "choices": [],
                          "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}))
    events.append((None, "[DONE]"))
    return httpx2.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})


def openai_error(status: int, code: str, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message, "type": code, "code": code}})


def anthropic_message(stop_reason: str = "end_turn", **extra: object) -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Es una medida del desorden."},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 9},
        **extra,
    })


def anthropic_stream(*deltas: str) -> httpx2.Response:
    events: list[tuple[str | None, object]] = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
    ]
    events += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": d}}) for d in deltas]
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 4}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return httpx2.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})


def anthropic_error(status: int, error_type: str, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"type": "error", "error": {"type": error_type, "message": message}})


# ----------------------------------------------------------------------
# Validación con Pydantic
# ----------------------------------------------------------------------

@pytest.mark.parametrize("temperature", [-0.1, 2.1, 3.5])
def test_temperature_fuera_de_rango(temperature):
    with pytest.raises(ValidationError):
        GenerationParams(temperature=temperature)


def test_parametros_desconocidos_y_max_tokens_invalido():
    with pytest.raises(ValidationError):
        GenerationParams(max_tokens=0)
    with pytest.raises(ValidationError):
        GenerationParams(top_k=5)


def test_conversacion_debe_terminar_en_usuario():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[ChatMessage(role=Role.SYSTEM, content="hola")])
    with pytest.raises(ValidationError):
        ChatRequest(messages=[
            ChatMessage(role=Role.USER, content="hola"),
            ChatMessage(role=Role.ASSISTANT, content="chau"),
        ])
    with pytest.raises(ValidationError):
        ChatMessage(role=Role.USER, content="")


def test_settings_exige_key_del_proveedor_y_la_oculta():
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(provider=Provider.ANTHROPIC, openai_api_key=SecretStr("sk-x"))
    settings = Settings(provider=Provider.OPENAI, openai_api_key=SecretStr("sk-secreta"))
    assert "sk-secreta" not in repr(settings)


def test_settings_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    settings = Settings.from_env()
    assert settings.provider is Provider.ANTHROPIC
    assert settings.model_for(settings.provider) == "claude-sonnet-5-5"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def test_factory_elige_cliente_segun_configuracion():
    openai_settings = Settings(provider="openai", openai_api_key=SecretStr("sk-x"))
    anthropic_settings = Settings(provider="anthropic", anthropic_api_key=SecretStr("sk-ant-x"))
    assert isinstance(AsyncLLMManager.from_settings(openai_settings).client, OpenAIClient)
    manager = AsyncLLMManager.from_settings(anthropic_settings)
    assert isinstance(manager.client, AnthropicClient)
    assert manager.model == "claude-opus-5-5"


# ----------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------

def test_openai_generate():
    recorder = Recorder(openai_completion())
    response = run(openai_client(recorder).generate(MESSAGES))

    assert response.ok
    assert response.content == "Es una medida del desorden."
    assert response.usage.output_tokens == 8
    body = recorder.body()
    assert body["messages"][0] == {"role": "system", "content": "Sos breve."}
    assert body["max_completion_tokens"] == 16_000
    assert "temperature" not in body  # sólo se envía si se pide


def test_openai_envia_temperature_si_se_pide():
    recorder = Recorder(openai_completion())
    run(openai_client(recorder).generate(MESSAGES, GenerationParams(temperature=0.2)))
    assert recorder.body()["temperature"] == 0.2


def test_openai_stream():
    recorder = Recorder(openai_stream("La ", "entropía ", "mide..."))
    chunks = run(collect(openai_client(recorder).stream(MESSAGES)))

    assert "".join(c.delta for c in chunks) == "La entropía mide..."
    final = chunks[-1]
    assert final.done and final.error is None
    assert final.finish_reason == "stop"
    assert final.usage.output_tokens == 3
    assert recorder.body()["stream"] is True


def test_openai_reintenta_rate_limit():
    recorder = Recorder(openai_error(429, "rate_limit_exceeded", "slow down"), openai_completion())
    response = run(openai_client(recorder).generate(MESSAGES))
    assert response.ok
    assert response.attempts == 2


def test_openai_sin_cuota_no_reintenta():
    recorder = Recorder(openai_error(429, "insufficient_quota", "no credits"))
    response = run(openai_client(recorder).generate(MESSAGES))
    assert response.error.kind is ErrorKind.RATE_LIMIT
    assert not response.error.retryable
    assert len(recorder.requests) == 1


def test_openai_api_key_invalida_devuelve_error_controlado():
    recorder = Recorder(openai_error(401, "invalid_api_key", "Incorrect API key"))
    response = run(openai_client(recorder).generate(MESSAGES))
    assert response.error.kind is ErrorKind.AUTHENTICATION
    assert response.error.status_code == 401
    assert len(recorder.requests) == 1


def test_openai_error_de_red_agota_reintentos():
    failures = [httpx2.ConnectError("sin red") for _ in range(3)]
    recorder = Recorder(*failures)
    response = run(openai_client(recorder).generate(MESSAGES, GenerationParams(max_retries=2)))
    assert response.error.kind is ErrorKind.CONNECTION
    assert response.attempts == 3


def test_openai_stream_reintenta_antes_del_primer_token():
    recorder = Recorder(openai_error(500, "server_error", "boom"), openai_stream("hola"))
    chunks = run(collect(openai_client(recorder).stream(MESSAGES)))
    assert "".join(c.delta for c in chunks) == "hola"
    assert chunks[-1].error is None


def test_openai_stream_error_controlado():
    recorder = Recorder(openai_error(401, "invalid_api_key", "Incorrect API key"))
    chunks = run(collect(openai_client(recorder).stream(MESSAGES)))
    assert len(chunks) == 1
    assert chunks[0].done and chunks[0].error.kind is ErrorKind.AUTHENTICATION


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------

def test_anthropic_generate():
    recorder = Recorder(anthropic_message())
    response = run(anthropic_client(recorder).generate(MESSAGES))

    assert response.ok
    assert response.content == "Es una medida del desorden."  # ignora el bloque de razonamiento
    assert response.finish_reason == "end_turn"
    body = recorder.body()
    assert body["system"] == "Sos breve."  # el system va aparte
    assert body["messages"] == [{"role": "user", "content": "¿Qué es la entropía?"}]


def test_anthropic_ignora_temperature():
    recorder = Recorder(anthropic_message())
    response = run(anthropic_client(recorder).generate(MESSAGES, GenerationParams(temperature=0.7)))
    assert response.ok
    assert "temperature" not in recorder.body()


def test_anthropic_stream():
    recorder = Recorder(anthropic_stream("La ", "entropía ", "mide..."))
    chunks = run(collect(anthropic_client(recorder).stream(MESSAGES)))

    assert "".join(c.delta for c in chunks) == "La entropía mide..."
    final = chunks[-1]
    assert final.done and final.error is None
    assert final.finish_reason == "end_turn"
    assert final.usage.output_tokens == 4


def test_anthropic_reintenta_sobrecarga():
    recorder = Recorder(anthropic_error(529, "overloaded_error", "Overloaded"), anthropic_message())
    response = run(anthropic_client(recorder).generate(MESSAGES))
    assert response.ok
    assert response.attempts == 2


def test_anthropic_respeta_retry_after():
    rate_limited = anthropic_error(429, "rate_limit_error", "slow down")
    rate_limited.headers["retry-after"] = "0"
    recorder = Recorder(rate_limited, anthropic_message())
    response = run(anthropic_client(recorder).generate(MESSAGES))
    assert response.ok and response.attempts == 2


def test_anthropic_request_invalido_no_reintenta():
    recorder = Recorder(anthropic_error(404, "not_found_error", "model: claude-x"))
    response = run(anthropic_client(recorder).generate(MESSAGES))
    assert response.error.kind is ErrorKind.INVALID_REQUEST
    assert len(recorder.requests) == 1


def test_anthropic_refusal():
    refusal = anthropic_message(
        "refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None}
    )
    response = run(anthropic_client(Recorder(refusal)).generate(MESSAGES))
    assert response.error.kind is ErrorKind.REFUSAL
    assert "cyber" in response.error.message


# ----------------------------------------------------------------------
# Lógica común de BaseLLMClient (timeouts, streaming cortado, semáforo)
# ----------------------------------------------------------------------

class FakeClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(self, delay: float = 0.0, fail_after_first_chunk: bool = False) -> None:
        super().__init__("fake", retry_base_delay=0)
        self.delay = delay
        self.fail_after_first_chunk = fail_after_first_chunk
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return ModelResponse(provider=self.provider, model=self.model, content="ok")

    async def _stream(self, messages: Messages, params: GenerationParams) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        yield StreamChunk(delta="hola ")
        if self.fail_after_first_chunk:
            raise ConnectionResetError("se cortó")
        yield StreamChunk(done=True, finish_reason="stop")

    def _classify_error(self, exc: Exception) -> LLMError:
        return LLMError(kind=ErrorKind.CONNECTION, message=str(exc), retryable=True)


def test_timeout_devuelve_error_controlado():
    client = FakeClient(delay=1.0)
    response = run(client.generate(MESSAGES, GenerationParams(timeout_s=0.05, max_retries=1)))
    assert response.error.kind is ErrorKind.TIMEOUT
    assert response.attempts == 2


def test_stream_cortado_a_mitad_no_se_reintenta():
    client = FakeClient(fail_after_first_chunk=True)
    chunks = run(collect(client.stream(MESSAGES)))
    assert [c.delta for c in chunks] == ["hola ", ""]
    assert chunks[-1].error.kind is ErrorKind.CONNECTION
    assert client.calls == 1  # reintentar duplicaría "hola " en pantalla


def test_semaforo_limita_concurrencia():
    client = FakeClient(delay=0.02)
    manager = AsyncLLMManager(client, max_concurrency=2)

    async def burst() -> list[ModelResponse]:
        return await asyncio.gather(*(manager.generate(f"pregunta {i}") for i in range(6)))

    responses = run(burst())
    assert all(r.ok for r in responses)
    assert client.max_active == 2


def test_manager_valida_antes_de_llamar():
    client = FakeClient()
    manager = AsyncLLMManager(client)
    with pytest.raises(ValidationError):
        run(manager.generate("hola", temperature=5))
    assert client.calls == 0
