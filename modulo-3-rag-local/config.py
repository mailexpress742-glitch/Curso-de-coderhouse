import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

from schemas import EmbeddingsProvider, Provider

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    provider: Provider = Provider.OPENAI
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-opus-5-5"

    # Con "local" los embeddings se calculan en la máquina y no hace falta ninguna key.
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.LOCAL
    openai_embedding_model: str = "text-embedding-3-small"

    data_dir: Path = BASE_DIR / "data"
    store_dir: Path = BASE_DIR / "vectorstore"
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, ge=1, le=8)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.api_key is None:
            raise ValueError(f"falta {self.provider.upper()}_API_KEY para usar {self.provider}")
        if self.embeddings_provider == EmbeddingsProvider.OPENAI and self.openai_api_key is None:
            raise ValueError("falta OPENAI_API_KEY para usar embeddings de OpenAI")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("el overlap tiene que ser menor que el tamaño de los fragmentos")
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
            "embeddings_provider": os.getenv("EMBEDDINGS_PROVIDER"),
            "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL"),
            "chunk_size": os.getenv("RAG_CHUNK_SIZE"),
            "chunk_overlap": os.getenv("RAG_CHUNK_OVERLAP"),
            "top_k": os.getenv("RAG_TOP_K"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
