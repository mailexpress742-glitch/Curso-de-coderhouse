import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    pinecone_api_key: SecretStr
    index_name: str = Field(default="rag-pydantic-docs", min_length=1)
    namespace: str = "pydantic-docs"
    cloud: str = "aws"
    region: str = "us-east-1"

    # Con "local" los embeddings se calculan en la máquina y no hace falta ninguna key.
    embeddings_provider: str = Field(default="local", pattern="^(local|openai)$")
    local_embedding_model: str = "jinaai/jina-embeddings-v2-base-es"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: SecretStr | None = None

    data_dir: Path = BASE_DIR / "data"
    chunk_size: int = Field(default=600, ge=100, le=2000)
    chunk_overlap: int = Field(default=60, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    # Peso de BM25 en la combinación; el de la búsqueda vectorial es el complemento.
    bm25_weight: float = Field(default=0.4, ge=0, le=1)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.embeddings_provider == "openai" and self.openai_api_key is None:
            raise ValueError("falta OPENAI_API_KEY para usar embeddings de OpenAI")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("el overlap tiene que ser menor que el tamaño de los fragmentos")
        return self

    @classmethod
    def from_env(cls) -> Self:
        values = {
            "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
            "index_name": os.getenv("INDEX_NAME"),
            "namespace": os.getenv("PINECONE_NAMESPACE"),
            "cloud": os.getenv("PINECONE_CLOUD"),
            "region": os.getenv("PINECONE_REGION"),
            "embeddings_provider": os.getenv("EMBEDDINGS_PROVIDER"),
            "local_embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL"),
            "openai_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "chunk_size": os.getenv("RAG_CHUNK_SIZE"),
            "chunk_overlap": os.getenv("RAG_CHUNK_OVERLAP"),
            "top_k": os.getenv("RAG_TOP_K"),
            "bm25_weight": os.getenv("BM25_WEIGHT"),
        }
        # Las variables vacías del .env cuentan como no definidas.
        return cls.model_validate({key: value for key, value in values.items() if value})
