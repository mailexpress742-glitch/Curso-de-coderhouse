"""Esquemas Pydantic compartidos por todos los proveedores.

Todo lo que entra (mensajes, parámetros) y todo lo que sale (respuestas, chunks
de streaming, errores) pasa por estos modelos. Así el resto del código nunca
trabaja con diccionarios anidados ni con los tipos propios de cada SDK.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación, con el mismo formato para cualquier proveedor."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


type Messages = Sequence[ChatMessage]


class GenerationParams(BaseModel):
    """Parámetros de una llamada al modelo, validados antes de tocar la red."""

    model_config = ConfigDict(extra="forbid")

    # None = no se envía y el modelo usa su valor por defecto. Algunos modelos
    # (los de razonamiento de OpenAI y los Claude actuales) rechazan valores
    # distintos al default, por eso es opcional.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int = Field(default=16_000, gt=0, le=128_000)
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)


class ChatRequest(BaseModel):
    """Mensajes + parámetros. Valida la forma de la conversación."""

    messages: list[ChatMessage] = Field(min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)

    @model_validator(mode="after")
    def _check_conversation(self) -> "ChatRequest":
        turns = [m for m in self.messages if m.role is not Role.SYSTEM]
        if not turns:
            raise ValueError("la conversación necesita al menos un mensaje de usuario")
        if turns[-1].role is not Role.USER:
            raise ValueError("el último mensaje debe ser del usuario")
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
    """Error controlado: se devuelve en lugar de propagar la excepción del SDK."""

    kind: ErrorKind
    message: str
    retryable: bool = False
    status_code: int | None = None


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelResponse(BaseModel):
    """Respuesta normalizada: mismo formato sin importar el proveedor."""

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
    """Un fragmento del streaming.

    Los chunks intermedios traen `delta` (el texto nuevo). El último trae
    `done=True` y, según cómo terminó, `finish_reason`, `usage` o `error`.
    """

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None
    error: LLMError | None = None
