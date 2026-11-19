import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-opus-5-5"

    # Redis guarda el estado de los jobs y los checkpoints de LangGraph. Hace falta Redis con
    # los módulos RediSearch y RedisJSON (la imagen redis/redis-stack-server los trae).
    redis_url: str = "redis://localhost:6379/0"
    job_ttl_s: int = Field(default=7 * 24 * 3600, ge=60)

    # Workers: cuántas tareas corren a la vez y cuánto puede tardar una.
    worker_concurrency: int = Field(default=5, ge=1, le=50)
    job_timeout_s: int = Field(default=300, ge=10)

    # Condiciones de parada del supervisor y techo de pasos del grafo.
    max_steps: int = Field(default=8, ge=3, le=30)
    max_attempts: int = Field(default=2, ge=1, le=5)
    recursion_limit: int = Field(default=40, ge=10, le=200)

    # Observabilidad (Arize Phoenix). Sin endpoint no se envían trazas.
    phoenix_endpoint: str | None = None
    phoenix_project: str = "orquestador-multiagente"

    @model_validator(mode="after")
    def api_key_required(self) -> Self:
        if self.api_key is None:
            raise ValueError(f"falta {self.provider.upper()}_API_KEY para usar {self.provider}")
        return self

    @property
    def api_key(self) -> SecretStr | None:
        return self.openai_api_key if self.provider == "openai" else self.anthropic_api_key

    @property
    def model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.anthropic_model

    @classmethod
    def from_env(cls) -> Self:
        values = {
            "provider": os.getenv("LLM_PROVIDER"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
            "redis_url": os.getenv("REDIS_URL"),
            "job_ttl_s": os.getenv("JOB_TTL_S"),
            "worker_concurrency": os.getenv("WORKER_CONCURRENCY"),
            "job_timeout_s": os.getenv("JOB_TIMEOUT_S"),
            "max_steps": os.getenv("SUPERVISOR_MAX_STEPS"),
            "max_attempts": os.getenv("SUPERVISOR_MAX_ATTEMPTS"),
            "recursion_limit": os.getenv("GRAPH_RECURSION_LIMIT"),
            "phoenix_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
            "phoenix_project": os.getenv("PHOENIX_PROJECT_NAME"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
