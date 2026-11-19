"""Lo que comparten los agentes: cómo se arma lo que ve cada uno y cómo se guardan sus aportes."""

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from state import Contribucion, DatoHerramienta


def text_of(message: BaseMessage) -> str:
    return str(message.text).strip()


def user_question(messages: list[BaseMessage]) -> str:
    """La pregunta original: el primer mensaje del usuario."""
    return next(text_of(m) for m in messages if isinstance(m, HumanMessage))


def extract_tool_data(messages: list[BaseMessage]) -> list[DatoHerramienta]:
    """Saca de la conversación interna de un agente las salidas crudas de sus herramientas."""
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = call["args"]

    data: list[DatoHerramienta] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                result: Any = json.loads(text_of(message))
            except json.JSONDecodeError:
                result = text_of(message)
            data.append(
                {
                    "herramienta": message.name or "desconocida",
                    "argumentos": calls.get(message.tool_call_id, {}),
                    "resultado": result,
                }
            )
    return data


def render_contributions(contributions: list[Contribucion]) -> str:
    """Los aportes previos como texto, con el autor de cada uno y sus datos completos."""
    if not contributions:
        return "(todavía no hay aportes de otros agentes)"
    blocks = []
    for item in contributions:
        block = f"### Aporte de [{item['agente']}] (intento {item['intento']})\n"
        block += f"Se le pidió: {item['instruccion']}\nRespondió: {item['resultado']}"
        if item["datos"]:
            data = json.dumps(item["datos"], ensure_ascii=False)
            block += f"\nDatos crudos de sus herramientas (JSON): {data}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_specialist_input(
    question: str, instruction: str, contributions: list[Contribucion]
) -> str:
    """Lo único que recibe un especialista: la pregunta, su tarea y los aportes de los demás.

    No recibe el historial de mensajes del sistema ni las decisiones del supervisor, para que
    no se contamine con información que no necesita.
    """
    return (
        f"Pregunta original del usuario:\n{question}\n\n"
        f"Tu tarea ahora:\n{instruction}\n\n"
        f"Aportes de otros agentes hasta ahora:\n{render_contributions(contributions)}"
    )
