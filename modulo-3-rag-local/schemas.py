from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingsProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class Fuente(BaseModel):
    """Fragmento de un documento que se usó para responder."""

    archivo: str
    fragmento: int = Field(ge=0, description="Posición del fragmento dentro del archivo.")
    extracto: str
    distancia: float = Field(description="Distancia coseno a la pregunta (menor = más parecido).")


class RespuestaLLM(BaseModel):
    """Lo que tiene que devolver el modelo de lenguaje."""

    respuesta: str = Field(min_length=1, description="Respuesta basada solo en el contexto.")
    encontrada: bool = Field(description="True si la respuesta está en el contexto.")
    fragmentos_usados: list[int] = Field(
        default_factory=list,
        description="Números de los fragmentos del contexto en los que se apoya la respuesta.",
    )

    @model_validator(mode="after")
    def found_answers_need_sources(self) -> Self:
        if self.encontrada and not self.fragmentos_usados:
            raise ValueError("una respuesta encontrada tiene que indicar qué fragmentos usó")
        return self


class RespuestaRAG(BaseModel):
    """Salida final del sistema: la respuesta y las fuentes de donde salió."""

    pregunta: str
    respuesta: str
    encontrada: bool
    fuentes: list[Fuente] = Field(default_factory=list)


class RAGError(Exception):
    """El sistema no pudo generar una respuesta válida."""


class EmbeddingMismatchError(Exception):
    """La base se creó con otro modelo de embeddings que el configurado."""
