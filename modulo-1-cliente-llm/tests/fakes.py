"""Respuestas HTTP simuladas para probar los clientes con los SDKs reales sin red."""

import json

import httpx2


class FakeTransport:
    """Devuelve las respuestas en orden y guarda los requests que recibió."""

    def __init__(self, *responses: httpx2.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def http_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))

    def sent_body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def sse(*events: tuple[str | None, dict | str]) -> httpx2.Response:
    lines = []
    for event, data in events:
        if event:
            lines.append(f"event: {event}")
        payload = data if isinstance(data, str) else json.dumps(data)
        lines.append(f"data: {payload}\n")
    body = "\n".join(lines) + "\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


def openai_completion(content: str = "Es una medida del desorden.") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "refusal": None},
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
    )


def openai_stream(*deltas: str) -> httpx2.Response:
    def chunk(choices: list, usage: dict | None = None) -> dict:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5-mini",
            "choices": choices,
            "usage": usage,
        }

    def choice(delta: dict, finish_reason: str | None = None) -> dict:
        return {"index": 0, "delta": delta, "finish_reason": finish_reason}

    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return sse(
        (None, chunk([choice({"role": "assistant", "content": ""})])),
        *[(None, chunk([choice({"content": delta})])) for delta in deltas],
        (None, chunk([choice({}, "stop")])),
        (None, chunk([], usage)),
        (None, "[DONE]"),
    )


def openai_error(status_code: int, code: str) -> httpx2.Response:
    return httpx2.Response(
        status_code,
        json={"error": {"message": f"error {code}", "type": code, "code": code}},
    )


def anthropic_message(stop_reason: str = "end_turn", **extra: object) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Es una medida del desorden."},
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 9},
            **extra,
        },
    )


def anthropic_stream(*deltas: str) -> httpx2.Response:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 1},
    }
    text_deltas = [
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": d}},
        )
        for d in deltas
    ]
    return sse(
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *text_deltas,
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


def anthropic_error(status_code: int, error_type: str) -> httpx2.Response:
    return httpx2.Response(
        status_code,
        json={"type": "error", "error": {"type": error_type, "message": f"error {error_type}"}},
    )
