import os
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

from schemas import Provider


class Settings(BaseModel):
    """Misma idea que el módulo 1: el proveedor y las keys salen de variables de entorno."""

    provider: Provider = Provider.OPENAI
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-opus-5-5"
    max_attempts: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def api_key_required(self) -> Self:
        if self.api_key is None:
            raise ValueError(f"falta {self.provider.upper()}_API_KEY para usar {self.provider}")
        return self

    @property
    def api_key(self) -> SecretStr | None:
        if self.provider == Provider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def model(self) -> str:
        if self.provider == Provider.OPENAI:
            return self.openai_model
        return self.anthropic_model

    @classmethod
    def from_env(cls, provider: str | None = None) -> Self:
        values = {
            "provider": provider or os.getenv("LLM_PROVIDER"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
            "max_attempts": os.getenv("PIPELINE_MAX_ATTEMPTS"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
