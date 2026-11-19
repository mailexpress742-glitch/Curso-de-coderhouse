"""Ejecuta el grafo y registra el flujo de delegación paso a paso."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def flow_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Convierte la actualización de un nodo en un evento legible del flujo."""
    if node == "supervisor":
        decision = update["decisiones"][-1]
        return {"nodo": "supervisor", **decision}

    contribution = update["contribuciones"][-1]
    return {
        "nodo": node,
        "intento": contribution["intento"],
        "instruccion": contribution["instruccion"],
        "herramientas": [dato["herramienta"] for dato in contribution["datos"]],
        "resultado": contribution["resultado"],
    }


async def run_question(
    graph: CompiledStateGraph, question: str, *, recursion_limit: int = 40
) -> dict[str, Any]:
    """Corre una consulta y devuelve la respuesta, el flujo de delegación y los aportes."""
    result: dict[str, Any] = {
        "pregunta": question,
        "flujo": [],
        "contribuciones": [],
        "respuesta_final": "",
        "completada": False,
        "error": None,
    }
    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(question)]},
            {"recursion_limit": recursion_limit},
            stream_mode="updates",
        ):
            for node, output in update.items():
                event = flow_event(node, output)
                result["flujo"].append(event)
                logger.info("%s", event)
                if node == "supervisor":
                    result["completada"] = output["task_completed"]
                else:
                    result["contribuciones"].extend(output["contribuciones"])
                    if "respuesta_final" in output:
                        result["respuesta_final"] = output["respuesta_final"]
    except GraphRecursionError:
        result["error"] = f"se superó el límite de {recursion_limit} pasos del grafo"
    return result
