"""Estado compartido del orquestador.

Todos los nodos leen y escriben este estado. Cada aporte de un especialista queda guardado
con el nombre del agente que lo hizo, así se sabe quién aportó qué y no se pierde contexto.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import MessagesState

# Agentes que ejecutan trabajo.
Especialista = Literal["investigador", "analista", "sintetizador", "ejecutor"]
# A dónde puede ir el flujo después del supervisor.
Destino = Literal["investigador", "analista", "sintetizador", "ejecutor", "FINISH"]

ESPECIALISTAS: tuple[Especialista, ...] = ("investigador", "analista", "sintetizador", "ejecutor")


class DatoHerramienta(TypedDict):
    """Salida cruda de una herramienta. Se guarda tal cual para que el siguiente agente la
    reciba sin pérdidas, en lugar de depender de que un modelo la resuma bien."""

    herramienta: str
    argumentos: dict[str, Any]
    resultado: Any


class Contribucion(TypedDict):
    """Un aporte de un agente: qué se le pidió, qué respondió y con qué datos."""

    agente: Especialista
    intento: int
    instruccion: str
    resultado: str
    datos: list[DatoHerramienta]


class Decision(TypedDict):
    """Registro de una decisión del supervisor (la bitácora del flujo)."""

    paso: int
    destino: Destino
    instruccion: str
    razon: str
    # Si una salvaguarda de código cambió lo que pidió el modelo, queda anotado acá.
    ajuste: str | None


class OrchestratorState(MessagesState):
    """Estado del grafo.

    `messages` (heredado) es la conversación compartida: la pregunta del usuario y un mensaje
    por cada intervención de un especialista, con su nombre en `name`.
    """

    next_agent: Destino
    instruccion: str
    # Con el reducer `operator.add` cada nodo agrega a la lista en vez de reemplazarla.
    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[Decision], operator.add]
    pasos: int
    respuesta_final: str
    task_completed: bool
    # Human-in-the-loop: la acción con efectos secundarios que espera aprobación y la respuesta.
    accion_propuesta: dict[str, Any] | None
    aprobacion: dict[str, Any] | None
    acciones_ejecutadas: Annotated[list[dict[str, Any]], operator.add]
