import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Resultado de una pregunta: la respuesta, la traza ReAct y cuántas veces usó herramientas."""

    pregunta: str
    respuesta: str
    pasos: list[dict[str, Any]] = field(default_factory=list)
    llamadas_a_herramientas: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def text_of(message: BaseMessage) -> str:
    return str(message.text).strip()


def parse_tool_output(message: ToolMessage) -> Any:
    try:
        return json.loads(text_of(message))
    except json.JSONDecodeError:
        return text_of(message)


def trace_step(number: int, node: str, message: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje del grafo en un paso legible de la traza."""
    if isinstance(message, AIMessage) and message.tool_calls:
        return {
            "paso": number,
            "nodo": node,
            "tipo": "accion",
            "razonamiento": text_of(message) or None,
            "herramientas": [
                {"nombre": call["name"], "argumentos": call["args"]} for call in message.tool_calls
            ],
        }
    if isinstance(message, ToolMessage):
        return {
            "paso": number,
            "nodo": node,
            "tipo": "observacion",
            "herramienta": message.name,
            "resultado": parse_tool_output(message),
        }
    return {"paso": number, "nodo": node, "tipo": "respuesta", "contenido": text_of(message)}


async def ask(
    graph: CompiledStateGraph, question: str, thread_id: str, *, recursion_limit: int = 12
) -> TurnResult:
    """Le hace una pregunta al agente y registra cada paso del ciclo.

    Con el mismo `thread_id` el checkpointer devuelve el historial de la conversación, así
    que el agente recuerda las preguntas anteriores. `recursion_limit` corta los bucles.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    result = TurnResult(pregunta=question, respuesta="")
    step = 0

    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(question)]}, config, stream_mode="updates"
        ):
            for node, output in update.items():
                for message in output["messages"]:
                    step += 1
                    entry = trace_step(step, node, message)
                    result.pasos.append(entry)
                    logger.info("paso %d [%s] %s", step, entry["tipo"], _summary(entry))
                    if entry["tipo"] == "accion":
                        result.llamadas_a_herramientas += len(entry["herramientas"])
                    elif entry["tipo"] == "respuesta":
                        result.respuesta = entry["contenido"]
    except GraphRecursionError:
        logger.error("el agente superó el límite de %d pasos", recursion_limit)
        result.error = f"se superó el límite de {recursion_limit} pasos sin llegar a una respuesta"
        result.respuesta = "No pude resolverlo en el número de pasos permitido."
    return result


def _summary(entry: dict[str, Any]) -> str:
    if entry["tipo"] == "accion":
        return ", ".join(f"{h['nombre']}({h['argumentos']})" for h in entry["herramientas"])
    if entry["tipo"] == "observacion":
        return f"{entry['herramienta']} -> {entry['resultado']}"
    return entry["contenido"][:120]
