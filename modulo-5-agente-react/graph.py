from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import Settings
from tools import TOOLS

SYSTEM_PROMPT = """Sos un asistente de soporte que responde preguntas sobre clientes y pedidos \
consultando una base de datos con herramientas.

- Usá las herramientas para obtener los datos. No inventes cifras, fechas ni productos.
- Si una pregunta necesita varios datos, pedilos uno por uno: el resultado de una herramienta \
puede darte el dato que necesita la siguiente.
- Si una herramienta devuelve un error, pensá por qué falló. Si podés corregirlo con otra \
herramienta (por ejemplo buscando el cliente por nombre), hacelo. Si no, pedile al usuario la \
aclaración que falta en vez de adivinar.
- Si el usuario hace referencia a algo de la conversación ("el último", "ese cliente"), \
resolvelo con los mensajes anteriores.
- Respondé en español, de forma breve, con los importes en pesos con separador de miles \
(por ejemplo $14.500)."""


class AgentState(MessagesState):
    """Estado del grafo: la lista de mensajes de la conversación.

    `MessagesState` ya define `messages` con el reducer `add_messages`: cada nodo devuelve
    solo los mensajes nuevos y LangGraph los agrega al historial sin pisarlo.
    """


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver[Any] | None,
    *,
    tools: Sequence[BaseTool] = TOOLS,
    max_context_messages: int = 30,
) -> CompiledStateGraph:
    """Arma el ciclo ReAct: modelo -> (¿pidió una herramienta?) -> herramientas -> modelo ..."""
    model_with_tools = model.bind_tools(list(tools))

    async def call_model(state: AgentState) -> dict[str, list[Any]]:
        # El estado guardado crece con cada turno. Al modelo solo le mandamos los últimos
        # mensajes, empezando siempre en un mensaje del usuario para no separar un pedido
        # de herramienta de su resultado.
        window = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=max_context_messages,
            start_on="human",
            include_system=False,
        )
        response = await model_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *window])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(list(tools)))
    builder.add_edge(START, "agent")
    # Si el último mensaje del modelo pide herramientas va a "tools", si no termina.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def build_model(settings: Settings) -> BaseChatModel:
    key = settings.api_key.get_secret_value()
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.model, api_key=key, max_tokens=2_000)
