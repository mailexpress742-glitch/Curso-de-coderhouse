from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import build_analyst_agent
from agents.common import build_specialist_input, extract_tool_data, text_of, user_question
from agents.research_agent import build_research_agent
from agents.supervisor import Limits, make_supervisor_node, route_from_supervisor
from agents.synthesis_agent import synthesize
from config import Settings
from state import Contribucion, Especialista, OrchestratorState

# Pasos máximos que puede dar un especialista por dentro (llamar herramientas y responder).
SPECIALIST_RECURSION_LIMIT = 12

Node = Callable[[OrchestratorState], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Models:
    """Un modelo por rol. Pueden ser el mismo, o modelos distintos según la tarea."""

    supervisor: BaseChatModel
    investigador: BaseChatModel
    analista: BaseChatModel
    sintetizador: BaseChatModel

    @classmethod
    def same(cls, model: BaseChatModel) -> "Models":
        return cls(model, model, model, model)


def contribution_update(
    agent: Especialista, state: OrchestratorState, result: str, data: list[Any]
) -> dict[str, Any]:
    """El aporte de un agente, con su nombre, y un mensaje en la conversación compartida."""
    previous = state.get("contribuciones", [])
    entry: Contribucion = {
        "agente": agent,
        "intento": 1 + sum(1 for item in previous if item["agente"] == agent),
        "instruccion": state["instruccion"],
        "resultado": result,
        "datos": data,
    }
    return {"contribuciones": [entry], "messages": [AIMessage(content=result, name=agent)]}


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
        return contribution_update(agent, state, text_of(messages[-1]), extract_tool_data(messages))

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


def build_graph(models: Models, limits: Limits | None = None) -> CompiledStateGraph:
    """Supervisor -> especialista -> supervisor -> ... -> FINISH.

    Los especialistas nunca hablan entre sí: todo pasa por el supervisor y por el estado.
    """
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", make_supervisor_node(models.supervisor, limits or Limits()))
    builder.add_node(
        "investigador",
        make_specialist_node("investigador", build_research_agent(models.investigador)),
    )
    builder.add_node(
        "analista", make_specialist_node("analista", build_analyst_agent(models.analista))
    )
    builder.add_node("sintetizador", make_synthesizer_node(models.sintetizador))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "investigador": "investigador",
            "analista": "analista",
            "sintetizador": "sintetizador",
            "FINISH": END,
        },
    )
    for agent in ("investigador", "analista", "sintetizador"):
        builder.add_edge(agent, "supervisor")
    return builder.compile()


def build_model(settings: Settings) -> BaseChatModel:
    key = settings.api_key.get_secret_value()
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.model, api_key=key, max_tokens=3_000)
