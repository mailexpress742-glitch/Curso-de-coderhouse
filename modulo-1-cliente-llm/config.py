"""Configuración del cliente, leída de variables de entorno (o de un archivo .env).

Las API keys se guardan como `SecretStr`: si la configuración aparece en un log
o en un traceback, se ve `**********` en lugar de la key.
"""

import os

from pydantic import BaseModel, Field, SecretStr, model_validator

from schemas import Provider

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-5-mini",
    Provider.ANTHROPIC: "claude-opus-5-5",
}

API_KEY_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


class Settings(BaseModel):
    provider: Provider = Provider.OPENAI
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = DEFAULT_MODELS[Provider.OPENAI]
    anthropic_model: str = DEFAULT_MODELS[Provider.ANTHROPIC]
    max_concurrency: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def _require_key_for_provider(self) -> "Settings":
        if self.api_key_for(self.provider) is None:
            raise ValueError(
                f"falta {API_KEY_ENV[self.provider]} para el proveedor '{self.provider}'"
            )
        return self

    def api_key_for(self, provider: Provider) -> SecretStr | None:
        return self.openai_api_key if provider is Provider.OPENAI else self.anthropic_api_key

    def model_for(self, provider: Provider) -> str:
        return self.openai_model if provider is Provider.OPENAI else self.anthropic_model

    @classmethod
    def from_env(cls, provider: Provider | str | None = None) -> "Settings":
        """Arma la configuración desde el entorno. `provider` pisa a LLM_PROVIDER."""
        values: dict[str, str] = {}
        env_map = {
            "provider": "LLM_PROVIDER",
            "openai_api_key": "OPENAI_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "openai_model": "OPENAI_MODEL",
            "anthropic_model": "ANTHROPIC_MODEL",
            "max_concurrency": "LLM_MAX_CONCURRENCY",
        }
        for field, env_name in env_map.items():
            value = os.getenv(env_name, "").strip()
            if value:
                values[field] = value
        if provider is not None:
            values["provider"] = provider
        return cls.model_validate(values)
