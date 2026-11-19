import asyncio
import time

import fakeredis
import pytest
from conftest import Env, decide, running_client, team, team_with_email_action, wait_for
from langchain_core.messages import AIMessage

from app.jobs import JobStatus, JobStore
from fakes import ScriptedModel, ScriptedSupervisor, answer

QUESTION = "Compará las reseñas de Aurora X y recomendame."


async def test_post_returns_a_job_id_immediately_without_waiting_for_the_result(env):
    async with running_client(env, team()) as client:
        response = await client.post("/tasks", json={"pregunta": QUESTION})
        assert response.status_code == 202
        body = response.json()
        assert body["status"] in {"PENDING", "RUNNING"}
        assert response.headers["location"] == f"/tasks/{body['job_id']}"


async def test_job_runs_in_the_background_and_ends_done(env):
    async with running_client(env, team()) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        job = await wait_for(client, job_id, "DONE", "FAILED")

    assert job["status"] == "DONE"
    assert job["resultado"] == "Recomiendo Aurora X."
    assert job["error"] is None
    nodes = [event["nodo"] for event in job["flujo"]]
    assert nodes[0] == "supervisor"
    assert {"investigador", "analista", "sintetizador"} <= set(nodes)


async def test_unknown_job_is_404(env):
    async with running_client(env, team()) as client:
        assert (await client.get("/tasks/no-existe")).status_code == 404


@pytest.mark.parametrize("body", [{}, {"pregunta": "hola"}, {"pregunta": "x" * 2001}])
async def test_invalid_requests_are_rejected_with_422(env, body):
    async with running_client(env, team()) as client:
        assert (await client.post("/tasks", json=body)).status_code == 422


async def test_health_checks_redis(env):
    async with running_client(env, team()) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}


async def test_the_api_stays_responsive_while_a_job_is_running(env):
    class SlowModel(ScriptedModel):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.sleep(0.4)
            return await super()._agenerate(*args, **kwargs)

    slow = SlowModel.with_script(AIMessage(content="Encontré datos."))
    async with running_client(env, team(investigador=slow)) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        started = time.perf_counter()
        for _ in range(5):
            assert (await client.get("/health")).status_code == 200
        assert time.perf_counter() - started < 0.3
        await wait_for(client, job_id, "DONE", "FAILED")


async def test_five_concurrent_requests_each_get_their_own_job(env):
    # Cada job usa el mismo equipo de modelos: el supervisor pasa directo a redactar.
    shared = team(
        supervisor=ScriptedSupervisor(decide("sintetizador", "Redactá."), decide("FINISH")),
    )
    async with running_client(env, shared) as client:
        responses = await asyncio.gather(
            *[client.post("/tasks", json={"pregunta": QUESTION}) for _ in range(5)]
        )
        ids = [r.json()["job_id"] for r in responses]
        assert len(set(ids)) == 5
        jobs = await asyncio.gather(*[wait_for(client, i, "DONE", "FAILED") for i in ids])
    assert all(job["status"] in {"DONE", "FAILED"} for job in jobs)


async def test_an_exception_in_the_background_marks_the_job_failed(env):
    class Broken(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("el proveedor no responde")

    broken = Broken.with_script(AIMessage(content="no se usa"))
    async with running_client(env, team(investigador=broken)) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        job = await wait_for(client, job_id, "DONE", "FAILED")

    assert job["status"] == "FAILED"
    assert "el proveedor no responde" in job["error"]
    assert job["resultado"] is None


async def test_a_job_that_takes_too_long_is_marked_failed(env):
    class Hanging(ScriptedModel):
        async def _agenerate(self, *args, **kwargs):
            await asyncio.sleep(5)

    hanging = Hanging.with_script(AIMessage(content="nunca"))
    async with running_client(env, team(investigador=hanging), timeout_s=1) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        job = await wait_for(client, job_id, "DONE", "FAILED", timeout=5)

    assert job["status"] == "FAILED"
    assert "tiempo máximo" in job["error"]


async def test_the_graph_recursion_limit_is_reported_as_failed(env):
    loop = ScriptedSupervisor(decide("investigador", "Buscá otra vez."))
    investigator = ScriptedModel.with_script(*[answer("Listo.") for _ in range(30)])
    models = team(supervisor=loop, investigador=investigator)
    async with running_client(env, models, recursion_limit=4) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        job = await wait_for(client, job_id, "DONE", "FAILED")

    assert job["status"] == "FAILED"
    assert "límite" in job["error"]


async def test_state_is_persisted_outside_the_process(env: Env):
    async with running_client(env, team()) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        await wait_for(client, job_id, "DONE")

    # "Reinicio": otra instancia de la API sobre el mismo Redis sigue viendo el job.
    async with running_client(env, team()) as client:
        job = (await client.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "DONE"
    assert job["resultado"] == "Recomiendo Aurora X."


async def test_unfinished_jobs_are_recovered_on_startup(env: Env):
    redis = fakeredis.FakeAsyncRedis(server=env.server, decode_responses=True)
    store = JobStore(redis)
    pending = await store.create(QUESTION)
    interrupted = await store.create(QUESTION)
    await store.update(interrupted.id, status=JobStatus.RUNNING)

    async with running_client(env, team()) as client:
        done = await wait_for(client, pending.id, "DONE", "FAILED")
        failed = (await client.get(f"/tasks/{interrupted.id}")).json()

    assert done["status"] == "DONE"
    assert failed["status"] == "FAILED"
    assert "reinició" in failed["error"]


# --- Human-in-the-loop ---


async def start_critical_job(client) -> str:
    response = await client.post(
        "/tasks", json={"pregunta": QUESTION + " Enviá el informe a ventas@ejemplo.com."}
    )
    return response.json()["job_id"]


async def test_a_critical_action_pauses_the_job_until_a_human_approves(env):
    async with running_client(env, team_with_email_action()) as client:
        job_id = await start_critical_job(client)
        job = await wait_for(client, job_id, "WAITING_APPROVAL", "DONE", "FAILED")

        assert job["status"] == "WAITING_APPROVAL"
        assert job["accion_pendiente"]["destinatario"] == "ventas@ejemplo.com"
        assert job["accion_pendiente"]["cuerpo"] == "Recomiendo Aurora X."
        # Mientras espera, el email NO salió.
        assert (await client.get("/outbox")).json() == []

        approval = await client.post(
            f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "ok"}
        )
        assert approval.status_code == 202
        job = await wait_for(client, job_id, "DONE", "FAILED")
        sent = (await client.get("/outbox")).json()

    assert job["status"] == "DONE"
    assert job["accion_pendiente"] is None
    assert len(sent) == 1
    assert sent[0]["destinatario"] == "ventas@ejemplo.com"
    assert sent[0]["cuerpo"] == "Recomiendo Aurora X."
    assert job["acciones_ejecutadas"][0]["accion"] == "enviar_informe_por_email"


async def test_a_rejected_action_is_never_executed(env):
    async with running_client(env, team_with_email_action()) as client:
        job_id = await start_critical_job(client)
        await wait_for(client, job_id, "WAITING_APPROVAL")
        await client.post(
            f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "no corresponde"}
        )
        job = await wait_for(client, job_id, "DONE", "FAILED")
        sent = (await client.get("/outbox")).json()

    assert job["status"] == "DONE"
    assert sent == []
    assert job["acciones_ejecutadas"] == []
    rejection = [e for e in job["flujo"] if e.get("aprobacion")]
    assert rejection[0]["aprobacion"]["aprobado"] is False


async def test_approving_a_job_that_is_not_waiting_returns_409(env):
    async with running_client(env, team()) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        await wait_for(client, job_id, "DONE")
        response = await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert response.status_code == 409


async def test_approving_twice_only_resumes_once(env):
    async with running_client(env, team_with_email_action()) as client:
        job_id = await start_critical_job(client)
        await wait_for(client, job_id, "WAITING_APPROVAL")
        first, second = await asyncio.gather(
            client.post(f"/tasks/{job_id}/approve", json={"aprobado": True}),
            client.post(f"/tasks/{job_id}/approve", json={"aprobado": True}),
        )
        await wait_for(client, job_id, "DONE", "FAILED")
        sent = (await client.get("/outbox")).json()

    assert sorted([first.status_code, second.status_code]) == [202, 409]
    assert len(sent) == 1


async def test_approving_an_unknown_job_is_404(env):
    async with running_client(env, team()) as client:
        response = await client.post("/tasks/nope/approve", json={"aprobado": True})
    assert response.status_code == 404


async def test_the_pause_survives_a_restart_of_the_api(env: Env):
    models = team_with_email_action()
    async with running_client(env, models) as client:
        job_id = await start_critical_job(client)
        await wait_for(client, job_id, "WAITING_APPROVAL")

    # La API se cae y vuelve a levantar: el job sigue esperando y se puede aprobar.
    async with running_client(env, models) as client:
        job = (await client.get(f"/tasks/{job_id}")).json()
        assert job["status"] == "WAITING_APPROVAL"
        await client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
        job = await wait_for(client, job_id, "DONE", "FAILED")
        sent = (await client.get("/outbox")).json()

    assert job["status"] == "DONE"
    assert len(sent) == 1
