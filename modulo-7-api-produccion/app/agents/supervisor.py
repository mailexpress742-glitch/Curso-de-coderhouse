"""El supervisor: decide quién interviene ahora o si la tarea está completa.

El modelo propone la decisión y unas salvaguardas en código la corrigen si hace falta: el
modelo puede equivocarse o entrar en un bucle de correcciones, el código no.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.agents.common import render_contributions, user_question
from app.state import ESPECIALISTAS, Contribucion, Decision, Destino, OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sos el supervisor de un equipo de agentes que responde consultas \
sobre productos. Tu trabajo es decidir quién interviene ahora, o si es momento de finalizar.

Equipo:
- investigador: busca en la base de productos (fichas, precios, reseñas). No analiza.
- analista: valida y analiza datos con herramientas de cálculo (sentimiento, estadísticas). \
No busca información nueva.
- sintetizador: redacta la respuesta final con los aportes de los demás. No tiene herramientas.
- ejecutor: realiza una acción con efectos secundarios, como enviar el informe por email. \
Elegilo SOLO si el usuario pidió esa acción de forma explícita y siempre después del \
sintetizador. Una persona tiene que aprobar la acción antes de que se ejecute.

Flujo esperado: primero el investigador reúne los datos, después el analista los procesa y \
el sintetizador redacta la respuesta. Si el usuario pidió una acción, el ejecutor la realiza \
al final. FINISH solo se elige cuando el sintetizador ya entregó un informe que cumple la \
rúbrica (y, si había una acción pedida, cuando el ejecutor ya intervino).

Rúbrica para validar cada resultado antes de seguir o finalizar:
1. Completitud: cubre todos los productos y todos los puntos de la pregunta.
2. Evidencia: cada cifra proviene de una herramienta, no de la memoria del agente.
3. Consistencia: los números del analista coinciden con los datos del investigador y el \
informe final no contradice a ninguno de los dos.
4. Pertinencia: el informe responde lo que se preguntó, con una recomendación si se pidió.

Si un resultado falla la rúbrica, enviá a ESE agente de nuevo con una instrucción concreta de \
qué corregir (máximo una corrección por agente). Si cumple, avanzá al siguiente paso. \
Cada instrucción tiene que ser autosuficiente: el agente solo ve la pregunta, tu instrucción y \
los aportes previos."""


class SupervisorDecision(BaseModel):
    """Salida estructurada que tiene que devolver el modelo supervisor."""

    siguiente: Destino = Field(description="Quién interviene ahora, o FINISH si está completo.")
    instruccion: str = Field(
        default="",
        description="Tarea concreta para el agente elegido. Vacía solo si siguiente es FINISH.",
    )
    razon: str = Field(min_length=3, description="Por qué se tomó esta decisión (rúbrica).")

    @model_validator(mode="after")
    def specialist_needs_instruction(self) -> "SupervisorDecision":
        if self.siguiente != "FINISH" and not self.instruccion.strip():
            raise ValueError("un especialista necesita una instrucción")
        return self


@dataclass(frozen=True)
class Limits:
    """Condiciones de parada: evitan el "supervisor infinito"."""

    max_steps: int = 8  # decisiones del supervisor por consulta
    max_attempts: int = 2  # veces que un mismo agente puede intervenir (1 corrección)


def attempts(contributions: list[Contribucion], agent: str) -> int:
    return sum(1 for item in contributions if item["agente"] == agent)


def build_context(state: OrchestratorState, limits: Limits) -> str:
    """Lo que ve el supervisor: la pregunta, los aportes y cuánto presupuesto queda."""
    contributions = state.get("contribuciones", [])
    steps = state.get("pasos", 0)
    summary = json.dumps(
        {agent: attempts(contributions, agent) for agent in ESPECIALISTAS}, ensure_ascii=False
    )
    return (
        f"Pregunta del usuario:\n{user_question(state['messages'])}\n\n"
        f"Intervenciones hasta ahora: {summary}\n"
        f"Decisiones usadas: {steps} de {limits.max_steps}. "
        f"Cada agente puede intervenir como máximo {limits.max_attempts} veces.\n\n"
        f"Aportes:\n{render_contributions(contributions)}\n\n"
        "¿Quién debe intervenir ahora, o es momento de finalizar?"
    )


CLOSING_TASK = (
    "Redactá la respuesta final para el usuario con los aportes disponibles. Si falta algo, "
    "aclaralo en la respuesta."
)


def closing_step(contributions: list[Contribucion]) -> tuple[Destino, str]:
    """Cómo cerrar la consulta: con el informe del sintetizador si falta, o directamente."""
    if attempts(contributions, "sintetizador") > 0:
        return "FINISH", ""
    return "sintetizador", CLOSING_TASK


def apply_guards(
    proposal: SupervisorDecision, contributions: list[Contribucion], steps: int, limits: Limits
) -> tuple[Destino, str, str | None]:
    """Corrige la propuesta del modelo si rompe una regla. Devuelve (destino, instrucción, ajuste).

    1. Sin presupuesto de pasos: se cierra (primero con la síntesis, si falta).
    2. Un agente que ya agotó sus intentos no puede volver a intervenir.
    3. No se puede finalizar sin que el sintetizador haya entregado un informe.
    """
    if steps >= limits.max_steps:
        destination, instruction = closing_step(contributions)
        return destination, instruction, f"se alcanzó el máximo de {limits.max_steps} decisiones"

    chosen = proposal.siguiente
    # Una acción con efectos secundarios se intenta una sola vez: no se pide aprobación dos veces.
    max_attempts = 1 if chosen == "ejecutor" else limits.max_attempts
    if chosen != "FINISH" and attempts(contributions, chosen) >= max_attempts:
        destination, instruction = closing_step(contributions)
        return destination, instruction, f"'{chosen}' ya agotó sus {max_attempts} intentos"

    if chosen == "ejecutor" and attempts(contributions, "sintetizador") == 0:
        return (
            "sintetizador",
            CLOSING_TASK,
            "la acción necesita primero el informe del sintetizador",
        )

    if chosen == "FINISH" and attempts(contributions, "sintetizador") == 0:
        return "sintetizador", CLOSING_TASK, "no se puede finalizar sin informe del sintetizador"

    return chosen, proposal.instruccion, None


def make_supervisor_node(model: BaseChatModel | Any, limits: Limits):
    """Crea el nodo supervisor. `model` necesita `with_structured_output`."""
    decider = model.with_structured_output(SupervisorDecision, method="json_schema").with_retry(
        stop_after_attempt=2
    )

    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        steps = state.get("pasos", 0)
        contributions = state.get("contribuciones", [])
        proposal: SupervisorDecision = await decider.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_context(state, limits))]
        )
        destination, instruction, adjustment = apply_guards(proposal, contributions, steps, limits)
        if adjustment:
            logger.warning("supervisor ajustado: %s", adjustment)

        record: Decision = {
            "paso": steps + 1,
            "destino": destination,
            "instruccion": instruction,
            "razon": proposal.razon,
            "ajuste": adjustment,
        }
        return {
            "next_agent": destination,
            "instruccion": instruction,
            "pasos": steps + 1,
            "decisiones": [record],
            "task_completed": destination == "FINISH",
        }

    return supervisor


def route_from_supervisor(state: OrchestratorState) -> Destino:
    """Arista condicional: el nodo siguiente es el que eligió el supervisor."""
    return state["next_agent"]
