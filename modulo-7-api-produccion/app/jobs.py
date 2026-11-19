"""Estado de los trabajos (jobs) en Redis.

Cada job es un hash `job:{id}` y una lista `job:{id}:flujo` con los pasos del grafo.
Con esto, cualquier proceso que lea Redis sabe en qué estado está una tarea.
"""

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import WatchError


class JobStatus(StrEnum):
    PENDING = "PENDING"  # en cola, todavía no empezó
    RUNNING = "RUNNING"  # un worker la está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausada: espera la decisión de una persona
    DONE = "DONE"
    FAILED = "FAILED"


class Job(BaseModel):
    id: str
    status: JobStatus
    pregunta: str
    creado: str
    actualizado: str
    resultado: str | None = None
    error: str | None = None
    # La acción que espera aprobación, mientras el estado es WAITING_APPROVAL.
    accion_pendiente: dict[str, Any] | None = None
    acciones_ejecutadas: list[dict[str, Any]] = Field(default_factory=list)
    flujo: list[dict[str, Any]] = Field(default_factory=list)


JSON_FIELDS = {"accion_pendiente", "acciones_ejecutadas"}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, redis: Redis, *, ttl_s: int = 7 * 24 * 3600) -> None:
        self.redis = redis
        self.ttl_s = ttl_s

    @staticmethod
    def key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, pregunta: str) -> Job:
        job_id = uuid.uuid4().hex
        stamp = now()
        await self.redis.hset(
            self.key(job_id),
            mapping={
                "id": job_id,
                "status": JobStatus.PENDING,
                "pregunta": pregunta,
                "creado": stamp,
                "actualizado": stamp,
            },
        )
        await self.redis.expire(self.key(job_id), self.ttl_s)
        return await self.get(job_id)  # type: ignore[return-value]

    async def get(self, job_id: str) -> Job | None:
        data = await self.redis.hgetall(self.key(job_id))
        if not data:
            return None
        for name in JSON_FIELDS & data.keys():
            data[name] = json.loads(data[name])
        flow = await self.redis.lrange(f"{self.key(job_id)}:flujo", 0, -1)
        return Job(**data, flujo=[json.loads(item) for item in flow])

    async def update(self, job_id: str, **fields: Any) -> None:
        mapping = {
            name: json.dumps(value, ensure_ascii=False) if name in JSON_FIELDS else str(value)
            for name, value in fields.items()
            if value is not None
        }
        mapping["actualizado"] = now()
        await self.redis.hset(self.key(job_id), mapping=mapping)
        # Un campo en None se borra (por ejemplo la acción pendiente al aprobarla).
        cleared = [name for name, value in fields.items() if value is None]
        if cleared:
            await self.redis.hdel(self.key(job_id), *cleared)

    async def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        key = f"{self.key(job_id)}:flujo"
        await self.redis.rpush(key, json.dumps(event, ensure_ascii=False, default=str))
        await self.redis.expire(key, self.ttl_s)

    async def transition(self, job_id: str, expected: JobStatus, new: JobStatus) -> bool:
        """Cambia el estado solo si todavía es `expected`. Evita que dos aprobaciones
        simultáneas reanuden el mismo job dos veces."""
        key = self.key(job_id)
        async with self.redis.pipeline() as pipe:
            try:
                await pipe.watch(key)
                if await pipe.hget(key, "status") != expected:
                    return False
                pipe.multi()
                pipe.hset(key, mapping={"status": new, "actualizado": now()})
                await pipe.execute()
                return True
            except WatchError:
                return False

    async def list_unfinished(self) -> list[Job]:
        """Jobs que quedaron sin terminar (para recuperarlos al arrancar la API)."""
        jobs = []
        async for key in self.redis.scan_iter(match="job:*"):
            if key.count(":") == 1:
                job = await self.get(key.split(":", 1)[1])
                if job and job.status in {JobStatus.PENDING, JobStatus.RUNNING}:
                    jobs.append(job)
        return jobs
