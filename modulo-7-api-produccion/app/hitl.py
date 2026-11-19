"""Human-in-the-loop: pausa obligatoria antes de ejecutar una acción con efectos secundarios.

Flujo: el supervisor elige "ejecutor" -> `proponer_accion` arma la acción exacta (a quién se
envía, con qué asunto y con qué cuerpo) -> `aprobacion_humana` PAUSA el grafo con `interrupt`
-> una persona aprueba o rechaza con POST /tasks/{id}/approve -> si aprobó, `ejecutar_accion`
la ejecuta. Si rechazó, no se ejecuta nada y el supervisor cierra la consulta.

Mientras espera, el estado queda guardado en el checkpointer de Redis: la pausa puede durar
minutos u horas y sobrevive a un reinicio de la API.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agents.common import contribution_update
from app.state import OrchestratorState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccionInfo:
    descripcion: str
    # Una acción crítica (efectos secundarios o costo alto) exige aprobación humana.
    critica: bool


ACCIONES: dict[str, AccionInfo] = {
    "enviar_informe_por_email": AccionInfo(
        descripcion="Envía el informe final por email a una persona externa.", critica=True
    ),
}


class PropuestaLLM(BaseModel):
    """Lo que decide el modelo: a quién se envía y con qué asunto. El cuerpo es el informe."""

    destinatario: str = Field(
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Email del destinatario."
    )
    asunto: str = Field(min_length=3, max_length=120)


PROPOSE_PROMPT = """Preparás el email con el que se envía el informe final. A partir de la \
instrucción, elegí el destinatario (un email que aparezca en la instrucción o en la pregunta \
del usuario; nunca inventes uno) y un asunto breve."""


class Outbox(Protocol):
    """Dónde "salen" los emails. En producción sería un servicio de correo."""

    async def send(self, destinatario: str, asunto: str, cuerpo: str) -> dict[str, Any]: ...


def make_propose_node(model: BaseChatModel):
    structured = model.with_structured_output(PropuestaLLM, method="json_schema")

    async def proponer_accion(state: OrchestratorState) -> dict[str, Any]:
        proposal: PropuestaLLM = await structured.ainvoke(
            [
                SystemMessage(PROPOSE_PROMPT),
                HumanMessage(
                    f"Instrucción: {state['instruccion']}\n\n"
                    f"Pregunta original: {state['messages'][0].content}"
                ),
            ]
        )
        action = {
            "nombre": "enviar_informe_por_email",
            "destinatario": proposal.destinatario,
            "asunto": proposal.asunto,
            # El cuerpo es el informe tal cual: el modelo no lo reescribe.
            "cuerpo": state["respuesta_final"],
        }
        return {"accion_propuesta": action, "aprobacion": None}

    return proponer_accion


async def aprobacion_humana(state: OrchestratorState) -> dict[str, Any]:
    """Pausa el grafo hasta recibir una decisión externa.

    `interrupt` guarda el estado y corta la ejecución. Cuando la API recibe la aprobación,
    reanuda el grafo con `Command(resume=...)` y `interrupt` devuelve esa decisión.
    """
    action = state["accion_propuesta"]
    if not ACCIONES[action["nombre"]].critica:
        return {"aprobacion": {"aprobado": True, "automatica": True, "comentario": ""}}

    decision: dict[str, Any] = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "motivo": ACCIONES[action["nombre"]].descripcion,
            "accion": action,
        }
    )
    approved = bool(decision.get("aprobado"))
    comment = str(decision.get("comentario", ""))
    update: dict[str, Any] = {
        "aprobacion": {"aprobado": approved, "automatica": False, "comentario": comment}
    }
    if not approved:
        # Queda registrado para que el supervisor lo vea y cierre la consulta.
        text = f"Acción rechazada por una persona: {comment or 'sin comentario'}."
        update.update(contribution_update("ejecutor", state, text, []))
    return update


def route_after_approval(state: OrchestratorState) -> Literal["ejecutar_accion", "supervisor"]:
    return "ejecutar_accion" if state["aprobacion"]["aprobado"] else "supervisor"


def make_execute_node(outbox: Outbox):
    async def ejecutar_accion(state: OrchestratorState) -> dict[str, Any]:
        action = state["accion_propuesta"]
        receipt = await outbox.send(action["destinatario"], action["asunto"], action["cuerpo"])
        logger.info("acción ejecutada: %s a %s", action["nombre"], action["destinatario"])
        text = (
            f"Email enviado a {action['destinatario']} (id {receipt['id']}) "
            f"con la aprobación de una persona."
        )
        return {
            **contribution_update("ejecutor", state, text, []),
            "acciones_ejecutadas": [{"accion": action["nombre"], **receipt}],
        }

    return ejecutar_accion


class MemoryOutbox:
    """Outbox en memoria, para tests y para correr sin Redis."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, destinatario: str, asunto: str, cuerpo: str) -> dict[str, Any]:
        receipt = {
            "id": uuid.uuid4().hex[:10],
            "destinatario": destinatario,
            "asunto": asunto,
            "enviado": datetime.now(UTC).isoformat(),
        }
        self.sent.append({**receipt, "cuerpo": cuerpo})
        return receipt
