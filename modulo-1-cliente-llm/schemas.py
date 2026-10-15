from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


type Messages = Sequence[ChatMessage]


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Es opcional porque varios modelos (los de razonamiento de OpenAI y los
    # Claude actuales) rechazan cualquier valor distinto al default.
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int = Field(default=16_000, gt=0, le=128_000)
    timeout_s: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)

    @model_validator(mode="after")
    def last_message_from_user(self) -> Self:
        turns = [m for m in self.messages if m.role != Role.SYSTEM]
        if not turns or turns[-1].role != Role.USER:
            raise ValueError("la conversación tiene que terminar con un mensaje del usuario")
        return self


class ErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    INVALID_REQUEST = "invalid_request"
    SERVER = "server"
    REFUSAL = "refusal"
    UNKNOWN = "unknown"


class LLMError(BaseModel):
    kind: ErrorKind
    message: str
    retryable: bool = False
    status_code: int | None = None


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    attempts: int = 1
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamChunk(BaseModel):
    """Fragmento de texto del stream. El último llega con done=True y trae
    el motivo de fin, el uso de tokens o el error."""

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None
    error: LLMError | None = None
