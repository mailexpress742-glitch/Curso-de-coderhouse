"""API REST del orquestador multi-agente.

POST /tasks              encola una tarea y devuelve un job_id (no espera el resultado)
GET  /tasks/{id}         estado del job: PENDING, RUNNING, WAITING_APPROVAL, DONE o FAILED
POST /tasks/{id}/approve aprueba o rechaza la acción crítica que dejó el job en pausa
GET  /outbox             emails "enviados" por el sistema (evidencia de las acciones aprobadas)
GET  /health             verifica la conexión con Redis
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, status
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.agents.supervisor import Limits
from app.config import Settings
from app.graph import Models, build_graph, build_model
from app.hitl import Outbox
from app.jobs import Job, JobStatus, JobStore
from app.observability import setup_observability
from app.outbox import RedisOutbox
from app.worker import JobRunner

logger = logging.getLogger(__name__)


class TaskCreate(BaseModel):
    pregunta: str = Field(min_length=5, max_length=2000)


class TaskAccepted(BaseModel):
    job_id: str
    status: JobStatus


class Approval(BaseModel):
    aprobado: bool
    comentario: str = Field(default="", max_length=500)


@dataclass
class Services:
    redis: Redis
    store: JobStore
    runner: JobRunner
    outbox: Outbox


ServicesFactory = Callable[[], AbstractAsyncContextManager[Services]]


def default_services(settings: Settings) -> ServicesFactory:
    """Servicios de producción: Redis para el estado y los checkpoints, y el modelo real."""

    @asynccontextmanager
    async def factory() -> AsyncIterator[Services]:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        store = JobStore(redis, ttl_s=settings.job_ttl_s)
        outbox = RedisOutbox(redis)
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
            await checkpointer.asetup()
            graph = build_graph(
                Models.same(build_model(settings)),
                outbox,
                checkpointer,
                Limits(max_steps=settings.max_steps, max_attempts=settings.max_attempts),
            )
            runner = JobRunner(
                graph,
                store,
                concurrency=settings.worker_concurrency,
                recursion_limit=settings.recursion_limit,
                timeout_s=settings.job_timeout_s,
            )
            await runner.start()
            try:
                yield Services(redis, store, runner, outbox)
            finally:
                await runner.stop()
                await redis.aclose()

    return factory


def create_app(services_factory: ServicesFactory | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        factory = services_factory
        if factory is None:
            load_dotenv()
            settings = Settings.from_env()
            setup_observability(settings)
            factory = default_services(settings)
        async with factory() as services:
            app.state.services = services
            yield

    # FastAPI trae trazado nativo de OpenTelemetry: con Phoenix configurado traza cada request
    # HTTP. El polling del cliente llenaría el dashboard de trazas irrelevantes, así que se
    # apaga; lo que interesa son las trazas del grafo (nodos y llamadas al modelo).
    app = FastAPI(
        title="Orquestador multi-agente",
        version="1.0",
        lifespan=lifespan,
        telemetry={"tracing": False},
    )

    def services(request: Request) -> Services:
        return request.app.state.services

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted)
    async def create_task(body: TaskCreate, request: Request, response: Response) -> TaskAccepted:
        svc = services(request)
        job = await svc.store.create(body.pregunta)
        await svc.runner.submit(job.id)
        response.headers["Location"] = f"/tasks/{job.id}"
        return TaskAccepted(job_id=job.id, status=job.status)

    @app.get("/tasks/{job_id}", response_model=Job)
    async def get_task(job_id: str, request: Request) -> Job:
        job = await services(request).store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no existe esa tarea")
        return job

    @app.post(
        "/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted
    )
    async def approve_task(job_id: str, body: Approval, request: Request) -> TaskAccepted:
        svc = services(request)
        if await svc.store.get(job_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no existe esa tarea")
        # El cambio de estado es atómico: dos aprobaciones a la vez no reanudan el job dos veces.
        moved = await svc.store.transition(job_id, JobStatus.WAITING_APPROVAL, JobStatus.PENDING)
        if not moved:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "la tarea no está esperando una aprobación"
            )
        await svc.runner.submit(job_id, {"aprobado": body.aprobado, "comentario": body.comentario})
        return TaskAccepted(job_id=job_id, status=JobStatus.PENDING)

    @app.get("/outbox")
    async def get_outbox(request: Request) -> list[dict[str, Any]]:
        return await services(request).outbox.list()  # type: ignore[attr-defined]

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        await services(request).redis.ping()
        return {"status": "ok"}

    return app


app = create_app()
