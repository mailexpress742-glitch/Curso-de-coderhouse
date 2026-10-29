import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from config import Settings
from schemas import EmbeddingMismatchError, EmbeddingsProvider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documentos"


@dataclass(frozen=True)
class Embedder:
    """Función de embeddings y un nombre para identificarla en la base."""

    function: embedding_functions.EmbeddingFunction
    name: str


def build_embedder(settings: Settings) -> Embedder:
    if settings.embeddings_provider == EmbeddingsProvider.OPENAI:
        function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key.get_secret_value(),
            model_name=settings.openai_embedding_model,
        )
        return Embedder(function, f"openai:{settings.openai_embedding_model}")

    # Modelo chico que corre en la máquina (ONNX). Se descarga la primera vez que se usa.
    return Embedder(embedding_functions.DefaultEmbeddingFunction(), "local:all-MiniLM-L6-v2")


def open_collection(store_dir: Path, embedder: Embedder, *, rebuild: bool = False) -> Collection:
    """Abre (o crea) la colección persistente.

    Guarda el nombre del modelo de embeddings en los metadatos de la colección y lo
    verifica al abrirla: consultar con un modelo distinto al de la indexación daría
    resultados sin sentido.
    """
    client = chromadb.PersistentClient(path=str(store_dir))

    if rebuild:
        existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            logger.info("colección eliminada para reconstruirla")

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=embedder.function,
        metadata={"hnsw:space": "cosine", "embedding_model": embedder.name},
    )

    stored = (collection.metadata or {}).get("embedding_model")
    if stored != embedder.name:
        raise EmbeddingMismatchError(
            f"la base se creó con '{stored}' y ahora se usa '{embedder.name}'. "
            "Reconstruila con: python ingest.py --rebuild"
        )
    return collection
