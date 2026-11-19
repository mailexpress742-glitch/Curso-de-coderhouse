"""Worker: ejecuta los jobs en segundo plano y deja el estado en Redis.

Los endpoints solo encolan y responden. Acá se corre el grafo, y cualquier excepción se
captura y se guarda como FAILED: si no, el cliente quedaría esperando para siempre.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.jobs import Job, JobStatus, JobStore
from app.observability import traced_job

logger = logging.getLogger(__name__)


def flow_event(node: str, update: dict[str, Any]) -> dict[str, Any] | None:
    """Convierte la actualización de un nodo en un evento legible del flujo."""
    if node == "supervisor":
        return {"nodo": node, **update["decisiones"][-1]}
    if node == "aprobacion_humana":
        return {"nodo": node, "aprobacion": update["aprobacion"]}
    if "contribuciones" in update:
        contribution = update["contribuciones"][-1]
        return {
            "nodo": node,
            "intento": contribution["intento"],
            "instruccion": contribution["instruccion"],
            "herramientas": [dato["herramienta"] for dato in contribution["datos"]],
            "resultado": contribution["resultado"],
        }
    if node == "proponer_accion":
        return {"nodo": node, "accion": update["accion_propuesta"]}
    return None


class JobRunner:
    """Cola en memoria con N workers asíncronos. El estado de cada job vive en Redis."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        store: JobStore,
        *,
        concurrency: int = 5,
        recursion_limit: int = 40,
        timeout_s: int = 300,
    ) -> None:
        self.graph = graph
        self.store = store
        self.concurrency = concurrency
        self.recursion_limit = recursion_limit
        self.timeout_s = timeout_s
        self._queue: asyncio.Queue[tuple[str, dict[str, Any] | None]] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._workers = [
            asyncio.create_task(self._work(), name=f"worker-{n}") for n in range(self.concurrency)
        ]
        await self._recover()

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def submit(self, job_id: str, resume: dict[str, Any] | None = None) -> None:
        """Encola un job. `resume` trae la decisión humana cuando se reanuda una pausa."""
        await self._queue.put((job_id, resume))

    async def _recover(self) -> None:
        """Al arrancar: reencola lo que no llegó a empezar y marca como FAILED lo que quedó
        a medias (la API se reinició mientras corría)."""
        for job in await self.store.list_unfinished():
            if job.status == JobStatus.PENDING:
                await self.submit(job.id)
            else:
                await self.store.update(
                    job.id, status=JobStatus.FAILED, error="la API se reinició durante la ejecución"
                )

    async def _work(self) -> None:
        while True:
            job_id, resume = await self._queue.get()
            try:
                await self.run(job_id, resume)
            finally:
                self._queue.task_done()

    async def join(self) -> None:
        """Espera a que se vacíe la cola (útil en tests)."""
        await self._queue.join()

    async def run(self, job_id: str, resume: dict[str, Any] | None = None) -> None:
        job = await self.store.get(job_id)
        if job is None:
            logger.error("el job %s no existe", job_id)
            return
        await self.store.update(job_id, status=JobStatus.RUNNING)
        kind = "reanudar" if resume is not None else "ejecutar"
        try:
            with traced_job(job_id, kind):
                async with asyncio.timeout(self.timeout_s):
                    await self._execute(job, resume)
        except TimeoutError:
            await self._fail(job_id, f"la tarea superó el tiempo máximo de {self.timeout_s} s")
        except GraphRecursionError:
            await self._fail(job_id, f"se superó el límite de {self.recursion_limit} pasos")
        except Exception as exc:
            logger.exception("falló el job %s", job_id)
            await self._fail(job_id, f"{type(exc).__name__}: {exc}")

    async def _fail(self, job_id: str, error: str) -> None:
        await self.store.update(job_id, status=JobStatus.FAILED, error=error)

    async def _execute(self, job: Job, resume: dict[str, Any] | None) -> None:
        config = {"configurable": {"thread_id": job.id}, "recursion_limit": self.recursion_limit}
        payload: Any = (
            Command(resume=resume)
            if resume is not None
            else {"messages": [HumanMessage(job.pregunta)]}
        )

        async for update in self.graph.astream(payload, config, stream_mode="updates"):
            if "__interrupt__" in update:
                request = update["__interrupt__"][0].value
                await self.store.update(
                    job.id, status=JobStatus.WAITING_APPROVAL, accion_pendiente=request["accion"]
                )
                return
            for node, output in update.items():
                event = flow_event(node, output)
                if event is not None:
                    await self.store.append_event(job.id, event)

        final = (await self.graph.aget_state(config)).values
        await self.store.update(
            job.id,
            status=JobStatus.DONE,
            resultado=final.get("respuesta_final", ""),
            acciones_ejecutadas=final.get("acciones_ejecutadas", []),
            accion_pendiente=None,
        )
