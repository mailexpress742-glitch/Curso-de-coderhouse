import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import fakeredis
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.supervisor import Limits, SupervisorDecision
from app.graph import Models, build_graph
from app.hitl import PropuestaLLM
from app.jobs import JobStore
from app.main import Services, create_app
from app.outbox import RedisOutbox
from app.worker import JobRunner
from fakes import ScriptedModel, ScriptedSupervisor, answer, call


def decide(siguiente: str, instruccion: str = "", razon: str = "motivo de prueba"):
    return SupervisorDecision(siguiente=siguiente, instruccion=instruccion, razon=razon)


def team(**overrides: Any) -> Models:
    """Un equipo de modelos falsos que investiga, analiza y redacta con herramientas reales."""
    parts: dict[str, Any] = {
        "supervisor": ScriptedSupervisor(
            decide("investigador", "Buscá las reseñas de Aurora X."),
            decide("analista", "Calculá el puntaje promedio."),
            decide("sintetizador", "Redactá la recomendación."),
            decide("FINISH"),
        ),
        "investigador": ScriptedModel.with_script(
            call("buscar_resenas", {"producto": "aurora"}), answer("Encontré 6 reseñas.")
        ),
        "analista": ScriptedModel.with_script(
            call("calcular_estadisticas", {"valores": [5, 4, 3]}), answer("Promedio 4,0.")
        ),
        "sintetizador": ScriptedModel.with_script(answer("Recomiendo Aurora X.")),
        "ejecutor": ScriptedSupervisor(
            PropuestaLLM(destinatario="ventas@ejemplo.com", asunto="Informe de reseñas")
        ),
    }
    return Models(**{**parts, **overrides})


def team_with_email_action() -> Models:
    return team(
        supervisor=ScriptedSupervisor(
            decide("investigador", "Buscá las reseñas de Aurora X."),
            decide("analista", "Calculá el puntaje promedio."),
            decide("sintetizador", "Redactá la recomendación."),
            decide("ejecutor", "Enviá el informe a ventas@ejemplo.com."),
            decide("FINISH"),
        )
    )


class Env:
    """Un entorno de prueba: Redis y checkpointer falsos que se reusan entre "reinicios"."""

    def __init__(self) -> None:
        self.server = fakeredis.FakeServer()
        self.checkpointer = InMemorySaver()

    def services(self, models: Models, **runner_options: Any):
        @asynccontextmanager
        async def factory() -> AsyncIterator[Services]:
            redis = fakeredis.FakeAsyncRedis(server=self.server, decode_responses=True)
            store = JobStore(redis)
            outbox = RedisOutbox(redis)
            graph = build_graph(models, outbox, self.checkpointer, Limits())
            options = {"concurrency": 5, "recursion_limit": 40, "timeout_s": 30, **runner_options}
            runner = JobRunner(graph, store, **options)
            await runner.start()
            try:
                yield Services(redis, store, runner, outbox)
            finally:
                await runner.stop()
                await redis.aclose()

        return factory


@asynccontextmanager
async def running_client(env: Env, models: Models, **runner_options: Any):
    """Un cliente HTTP contra la API con el ciclo de vida (lifespan) ya iniciado."""
    app = create_app(env.services(models, **runner_options))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def wait_for(client: httpx.AsyncClient, job_id: str, *statuses: str, timeout: float = 10):
    """Consulta el job hasta que llegue a alguno de los estados (como haría un cliente)."""
    async with asyncio.timeout(timeout):
        while True:
            body = (await client.get(f"/tasks/{job_id}")).json()
            if body["status"] in statuses:
                return body
            await asyncio.sleep(0.02)


@pytest.fixture
def env() -> Env:
    return Env()
