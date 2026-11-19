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

    # Condiciones de parada del supervisor.
    max_steps: int = Field(default=8, ge=3, le=30)
    max_attempts: int = Field(default=2, ge=1, le=5)
    # Techo de pasos del grafo completo: red de seguridad por si falla lo anterior.
    recursion_limit: int = Field(default=40, ge=10, le=200)

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
            "max_steps": os.getenv("SUPERVISOR_MAX_STEPS"),
            "max_attempts": os.getenv("SUPERVISOR_MAX_ATTEMPTS"),
            "recursion_limit": os.getenv("GRAPH_RECURSION_LIMIT"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
