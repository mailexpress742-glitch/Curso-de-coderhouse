"""El orquestador multi-agente del módulo 6, con persistencia en Redis y un paso de aprobación.

START -> supervisor --> investigador | analista | sintetizador --> supervisor
                    \\-> proponer_accion -> aprobacion_humana -> ejecutar_accion -> supervisor
                    \\-> FINISH -> END
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analyst_agent import build_analyst_agent
from app.agents.common import (
    build_specialist_input,
    contribution_update,
    extract_tool_data,
    text_of,
    user_question,
)
from app.agents.research_agent import build_research_agent
from app.agents.supervisor import Limits, make_supervisor_node, route_from_supervisor
from app.agents.synthesis_agent import synthesize
from app.config import Settings
from app.hitl import (
    Outbox,
    aprobacion_humana,
    make_execute_node,
    make_propose_node,
    route_after_approval,
)
from app.state import Especialista, OrchestratorState

SPECIALIST_RECURSION_LIMIT = 12

Node = Callable[[OrchestratorState], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Models:
    """Un modelo por rol. Pueden ser el mismo, o modelos distintos según la tarea."""

    supervisor: BaseChatModel
    investigador: BaseChatModel
    analista: BaseChatModel
    sintetizador: BaseChatModel
    ejecutor: BaseChatModel

    @classmethod
    def same(cls, model: BaseChatModel) -> "Models":
        return cls(model, model, model, model, model)


def make_specialist_node(agent: Especialista, graph: CompiledStateGraph) -> Node:
    """Nodo que ejecuta un agente con herramientas y le entrega solo lo que necesita."""

    async def node(state: OrchestratorState) -> dict[str, Any]:
        prompt = build_specialist_input(
            user_question(state["messages"]), state["instruccion"], state.get("contribuciones", [])
        )
        result = await graph.ainvoke(
            {"messages": [HumanMessage(prompt)]},
            {"recursion_limit": SPECIALIST_RECURSION_LIMIT},
        )
        messages = result["messages"]
        text = text_of(messages[-1])
        return contribution_update(agent, state, text, extract_tool_data(messages))

    return node


def make_synthesizer_node(model: BaseChatModel) -> Node:
    async def node(state: OrchestratorState) -> dict[str, Any]:
        report = await synthesize(
            model,
            user_question(state["messages"]),
            state["instruccion"],
            state.get("contribuciones", []),
        )
        return {**contribution_update("sintetizador", state, report, []), "respuesta_final": report}

    return node


def build_graph(
    models: Models,
    outbox: Outbox,
    checkpointer: BaseCheckpointSaver[Any] | None,
    limits: Limits | None = None,
) -> CompiledStateGraph:
    """Arma el grafo. El checkpointer guarda el estado de cada thread_id (un job por hilo)."""
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", make_supervisor_node(models.supervisor, limits or Limits()))
    research = build_research_agent(models.investigador)
    builder.add_node("investigador", make_specialist_node("investigador", research))
    builder.add_node(
        "analista", make_specialist_node("analista", build_analyst_agent(models.analista))
    )
    builder.add_node("sintetizador", make_synthesizer_node(models.sintetizador))
    builder.add_node("proponer_accion", make_propose_node(models.ejecutor))
    builder.add_node("aprobacion_humana", aprobacion_humana)
    builder.add_node("ejecutar_accion", make_execute_node(outbox))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "investigador": "investigador",
            "analista": "analista",
            "sintetizador": "sintetizador",
            "ejecutor": "proponer_accion",
            "FINISH": END,
        },
    )
    for agent in ("investigador", "analista", "sintetizador"):
        builder.add_edge(agent, "supervisor")
    builder.add_edge("proponer_accion", "aprobacion_humana")
    builder.add_conditional_edges(
        "aprobacion_humana",
        route_after_approval,
        {"ejecutar_accion": "ejecutar_accion", "supervisor": "supervisor"},
    )
    builder.add_edge("ejecutar_accion", "supervisor")
    return builder.compile(checkpointer=checkpointer)


def build_model(settings: Settings) -> BaseChatModel:
    key = settings.api_key.get_secret_value()
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.model, api_key=key, max_tokens=3_000)
