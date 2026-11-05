import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-opus-5-5"

    # Archivo donde el checkpointer guarda el estado de cada conversación (thread_id).
    db_path: Path = BASE_DIR / "checkpoints.sqlite"
    # Techo de pasos del grafo por consulta: corta los bucles antes de que cuesten dinero.
    recursion_limit: int = Field(default=12, ge=3, le=50)
    # Mensajes que se le mandan al modelo como máximo; el estado guardado conserva todo.
    max_context_messages: int = Field(default=30, ge=4)

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
    def from_env(cls, provider: str | None = None) -> Self:
        values = {
            "provider": provider or os.getenv("LLM_PROVIDER"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
            "db_path": os.getenv("CHECKPOINT_DB"),
            "recursion_limit": os.getenv("AGENT_RECURSION_LIMIT"),
            "max_context_messages": os.getenv("AGENT_MAX_CONTEXT_MESSAGES"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
