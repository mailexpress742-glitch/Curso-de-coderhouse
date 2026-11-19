import json
import uuid
from typing import Any

from redis.asyncio import Redis

from app.jobs import now

OUTBOX_KEY = "outbox"


class RedisOutbox:
    """Guarda en Redis los emails "enviados". Simula un servicio de correo."""

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def send(self, destinatario: str, asunto: str, cuerpo: str) -> dict[str, Any]:
        receipt = {
            "id": uuid.uuid4().hex[:10],
            "destinatario": destinatario,
            "asunto": asunto,
            "enviado": now(),
        }
        await self.redis.rpush(OUTBOX_KEY, json.dumps({**receipt, "cuerpo": cuerpo}))
        return receipt

    async def list(self) -> list[dict[str, Any]]:
        return [json.loads(item) for item in await self.redis.lrange(OUTBOX_KEY, 0, -1)]
