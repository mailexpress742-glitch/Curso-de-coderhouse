import pytest
from pydantic import ValidationError

from agents.supervisor import Limits, SupervisorDecision, apply_guards
from fakes import ScriptedModel, ScriptedSupervisor, answer, call
from graph import Models, build_graph
from runner import run_question
from state import Contribucion

QUESTION = "Compará las reseñas de Aurora X y Brisa Mini y recomendame uno."


def decide(siguiente: str, instruccion: str = "", razon: str = "motivo de prueba"):
    return SupervisorDecision(siguiente=siguiente, instruccion=instruccion, razon=razon)


def contribution(agent: str, attempt: int = 1) -> Contribucion:
    return {
        "agente": agent,
        "intento": attempt,
        "instruccion": "x",
        "resultado": "y",
        "datos": [],
    }


def specialists() -> dict[str, ScriptedModel]:
    """Modelos de los tres especialistas con un guion que usa una herramienta cada uno."""
    return {
        "investigador": ScriptedModel.with_script(
            call("buscar_resenas", {"producto": "aurora"}),
            answer("Encontré 6 reseñas de Aurora X."),
        ),
        "analista": ScriptedModel.with_script(
            call("calcular_estadisticas", {"valores": [5, 4, 3]}),
            answer("El promedio de puntaje es 4,0."),
        ),
        "sintetizador": ScriptedModel.with_script(answer("Recomiendo Aurora X (puntaje 4,0).")),
    }


def make_graph(supervisor: ScriptedSupervisor, limits: Limits | None = None, **overrides):
    team = {**specialists(), **overrides}
    models = Models(
        supervisor=supervisor,
        investigador=team["investigador"],
        analista=team["analista"],
        sintetizador=team["sintetizador"],
    )
    return build_graph(models, limits), team


HAPPY_PATH = (
    decide("investigador", "Buscá las reseñas de Aurora X."),
    decide("analista", "Calculá el puntaje promedio."),
    decide("sintetizador", "Redactá la recomendación."),
    decide("FINISH"),
)


# --- Salvaguardas del supervisor (funciones puras) ---


def test_proposal_without_instruction_is_invalid():
    with pytest.raises(ValidationError):
        SupervisorDecision(siguiente="analista", instruccion="  ", razon="porque sí")


def test_finish_does_not_need_an_instruction():
    assert decide("FINISH").siguiente == "FINISH"


def test_guard_lets_a_valid_decision_through():
    result = apply_guards(decide("analista", "hacé X"), [contribution("investigador")], 1, Limits())
    assert result == ("analista", "hacé X", None)


def test_guard_blocks_finish_without_a_report():
    destination, _, adjustment = apply_guards(decide("FINISH"), [], 1, Limits())
    assert destination == "sintetizador"
    assert "informe" in adjustment


def test_guard_allows_finish_after_the_report():
    result = apply_guards(decide("FINISH"), [contribution("sintetizador")], 3, Limits())
    assert result[0] == "FINISH"
    assert result[2] is None


def test_guard_stops_an_agent_that_exhausted_its_attempts():
    done = [contribution("analista", 1), contribution("analista", 2)]
    destination, _, adjustment = apply_guards(
        decide("analista", "otra vez"), done, 4, Limits(max_attempts=2)
    )
    assert destination == "sintetizador"
    assert "agotó" in adjustment


def test_guard_closes_when_the_step_budget_is_over():
    destination, _, adjustment = apply_guards(
        decide("investigador", "más"), [contribution("sintetizador")], 8, Limits(max_steps=8)
    )
    assert destination == "FINISH"
    assert "máximo" in adjustment


# --- El grafo completo ---


async def test_happy_path_delegates_in_order_and_finishes():
    graph, _ = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    result = await run_question(graph, QUESTION)

    nodes = [event["nodo"] for event in result["flujo"]]
    assert nodes == [
        "supervisor",
        "investigador",
        "supervisor",
        "analista",
        "supervisor",
        "sintetizador",
        "supervisor",
    ]
    assert result["completada"] is True
    assert result["error"] is None
    assert result["respuesta_final"] == "Recomiendo Aurora X (puntaje 4,0)."


async def test_each_contribution_is_attributed_to_its_agent():
    graph, _ = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    result = await run_question(graph, QUESTION)
    assert [c["agente"] for c in result["contribuciones"]] == [
        "investigador",
        "analista",
        "sintetizador",
    ]
    assert [c["intento"] for c in result["contribuciones"]] == [1, 1, 1]


async def test_raw_tool_data_is_kept_in_the_shared_state():
    graph, _ = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    result = await run_question(graph, QUESTION)
    research = result["contribuciones"][0]
    assert research["datos"][0]["herramienta"] == "buscar_resenas"
    assert research["datos"][0]["resultado"]["cantidad"] == 6


async def test_the_analyst_receives_the_researchers_raw_data():
    graph, team = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    await run_question(graph, QUESTION)
    analyst_input = str(team["analista"].seen[0])
    assert "AUR-1" in analyst_input  # una reseña cruda, no un resumen
    assert "Calculá el puntaje promedio." in analyst_input


async def test_a_specialist_does_not_see_the_supervisors_reasoning():
    graph, team = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    await run_question(graph, QUESTION)
    analyst_input = str(team["analista"].seen[0])
    assert "motivo de prueba" not in analyst_input
    # Solo el prompt del rol y un mensaje del usuario con la tarea.
    assert [m.type for m in team["analista"].seen[0]] == ["system", "human"]


async def test_supervisor_can_send_an_agent_back_to_refine():
    supervisor = ScriptedSupervisor(
        decide("investigador", "Buscá las reseñas."),
        decide("analista", "Calculá el promedio."),
        decide("analista", "Corregí: validá las reseñas antes de calcular."),
        decide("sintetizador", "Redactá."),
        decide("FINISH"),
    )
    analyst = ScriptedModel.with_script(
        answer("Promedio 4,0."), answer("Promedio 4,0, con reseñas validadas.")
    )
    graph, _ = make_graph(supervisor, analista=analyst)
    result = await run_question(graph, QUESTION)

    analyst_turns = [c for c in result["contribuciones"] if c["agente"] == "analista"]
    assert [c["intento"] for c in analyst_turns] == [1, 2]
    assert result["completada"]


async def test_finishing_early_still_produces_a_final_report():
    supervisor = ScriptedSupervisor(decide("investigador", "Buscá."), decide("FINISH"))
    graph, _ = make_graph(supervisor)
    result = await run_question(graph, QUESTION)

    adjusted = [e for e in result["flujo"] if e["nodo"] == "supervisor" and e["ajuste"]]
    assert adjusted[0]["destino"] == "sintetizador"
    assert result["respuesta_final"] == "Recomiendo Aurora X (puntaje 4,0)."


async def test_a_stuck_supervisor_is_cut_off_and_the_graph_still_ends():
    # El supervisor pide siempre "investigador": nunca va a decidir terminar.
    supervisor = ScriptedSupervisor(decide("investigador", "Buscá de nuevo."))
    investigator = ScriptedModel.with_script(*[answer("Listo.") for _ in range(10)])
    graph, _ = make_graph(
        supervisor, Limits(max_steps=4, max_attempts=2), investigador=investigator
    )
    result = await run_question(graph, QUESTION)

    decisions = [e for e in result["flujo"] if e["nodo"] == "supervisor"]
    assert len(decisions) <= 5
    assert result["completada"] is True
    assert result["error"] is None
    assert any(e["ajuste"] for e in decisions)
    assert result["respuesta_final"]


async def test_graph_recursion_limit_is_reported_without_raising():
    graph, _ = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    result = await run_question(graph, QUESTION, recursion_limit=3)
    assert "límite" in result["error"]
    assert result["completada"] is False


def test_graph_topology_has_the_supervisor_and_three_specialists():
    graph, _ = make_graph(ScriptedSupervisor(*HAPPY_PATH))
    nodes = set(graph.get_graph().nodes)
    assert {"supervisor", "investigador", "analista", "sintetizador"} <= nodes
