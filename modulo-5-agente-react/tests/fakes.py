from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field


def call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """Respuesta del modelo pidiendo usar una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def answer(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedModel(GenericFakeChatModel):
    """Modelo falso que devuelve respuestas de un guion, en orden, y guarda lo que recibió.

    Sirve para probar el grafo sin usar la API: el guion decide cuándo pide herramientas.
    """

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @classmethod
    def with_script(cls, *script: AIMessage) -> "ScriptedModel":
        return cls(messages=iter(script))

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)
