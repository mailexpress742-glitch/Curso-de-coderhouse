from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import MessagesState

from agent import ask
from fakes import ScriptedModel, answer, call
from graph import AgentState, build_graph


def graph_with(model: ScriptedModel, checkpointer=None, **kwargs):
    return build_graph(model, checkpointer or InMemorySaver(), **kwargs)


def test_state_inherits_from_messages_state():
    # TypedDict no soporta issubclass: se verifica la herencia por sus bases y sus campos.
    assert MessagesState in AgentState.__orig_bases__
    assert "messages" in AgentState.__annotations__


def test_graph_has_a_model_node_and_a_tools_node():
    graph = graph_with(ScriptedModel.with_script(answer("hola")))
    nodes = set(graph.get_graph().nodes)
    assert {"agent", "tools"} <= nodes


async def test_agent_answers_without_tools_when_it_does_not_need_them():
    model = ScriptedModel.with_script(answer("Hola, ¿en qué te ayudo?"))
    result = await ask(graph_with(model), "Hola", "t1")
    assert result.respuesta == "Hola, ¿en qué te ayudo?"
    assert result.llamadas_a_herramientas == 0
    assert [p["tipo"] for p in result.pasos] == ["respuesta"]


async def test_agent_chains_two_tool_calls_before_answering():
    model = ScriptedModel.with_script(
        call("buscar_cliente", {"nombre": "Ana"}),
        call("buscar_pedidos", {"cliente_id": 104}, "call_2"),
        answer("Ana gastó $11.600."),
    )
    result = await ask(graph_with(model), "¿Cuánto gastó Ana?", "t1")
    assert result.llamadas_a_herramientas == 2
    assert result.respuesta == "Ana gastó $11.600."
    assert [p["tipo"] for p in result.pasos] == [
        "accion",
        "observacion",
        "accion",
        "observacion",
        "respuesta",
    ]
    assert result.pasos[3]["resultado"]["total"] == 11600


async def test_tool_error_goes_back_to_the_model_which_tries_again():
    model = ScriptedModel.with_script(
        call("buscar_pedidos", {"cliente_id": 999}),
        call("buscar_cliente", {"nombre": "Lucía"}, "call_2"),
        answer("Probé con el nombre y encontré al cliente 102."),
    )
    result = await ask(graph_with(model), "pedidos del cliente 999", "t1")

    # Antes del segundo intento, el modelo recibió el error de la primera herramienta.
    second_call_input = model.seen[1]
    errors = [m for m in second_call_input if isinstance(m, ToolMessage)]
    assert "no existe el cliente 999" in errors[0].content
    assert result.llamadas_a_herramientas == 2
    assert result.error is None


async def test_recursion_limit_stops_a_loop_without_raising():
    # El modelo pide una herramienta una y otra vez.
    loop = [call("buscar_pedidos", {"cliente_id": 102}, f"c{i}") for i in range(30)]
    model = ScriptedModel.with_script(*loop)
    result = await ask(graph_with(model), "pedidos", "t1", recursion_limit=5)
    assert result.error is not None
    assert "5 pasos" in result.error
    assert result.llamadas_a_herramientas < 5


async def test_same_thread_remembers_previous_questions():
    model = ScriptedModel.with_script(answer("Son 3 pedidos."), answer("El último es P-2003."))
    graph = graph_with(model)
    await ask(graph, "¿Cuántos pedidos tuvo el 102?", "hilo-a")
    await ask(graph, "¿Y el último?", "hilo-a")

    texts = [m.content for m in model.seen[1] if isinstance(m, HumanMessage)]
    assert texts == ["¿Cuántos pedidos tuvo el 102?", "¿Y el último?"]


async def test_a_different_thread_starts_with_a_clean_history():
    model = ScriptedModel.with_script(answer("Son 3 pedidos."), answer("No sé a qué te referís."))
    graph = graph_with(model)
    await ask(graph, "¿Cuántos pedidos tuvo el 102?", "hilo-a")
    await ask(graph, "¿Y el último?", "hilo-b")

    texts = [m.content for m in model.seen[1] if isinstance(m, HumanMessage)]
    assert texts == ["¿Y el último?"]


async def test_memory_survives_closing_and_reopening_the_sqlite_file(tmp_path):
    db = str(tmp_path / "memoria.sqlite")

    first = ScriptedModel.with_script(answer("Son 3 pedidos."))
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await ask(graph_with(first, saver), "¿Cuántos pedidos tuvo el 102?", "hilo-a")

    # Otra conexión, como si el programa se hubiera reiniciado.
    second = ScriptedModel.with_script(answer("El último es P-2003."))
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await ask(graph_with(second, saver), "¿Y el último?", "hilo-a")

    texts = [m.content for m in second.seen[0] if isinstance(m, HumanMessage)]
    assert texts == ["¿Cuántos pedidos tuvo el 102?", "¿Y el último?"]


async def test_old_messages_are_trimmed_before_calling_the_model():
    script = [answer(f"respuesta {i}") for i in range(6)]
    model = ScriptedModel.with_script(*script)
    graph = graph_with(model, max_context_messages=4)
    for i in range(6):
        await ask(graph, f"pregunta {i}", "t1")

    last_input = [m for m in model.seen[-1] if m.type != "system"]
    assert len(last_input) <= 4
    assert isinstance(last_input[0], HumanMessage)
    assert last_input[-1].content == "pregunta 5"
