"""Procesa los documentos de data/ y los sube al índice de Pinecone.

python ingest.py

Es idempotente: solo calcula embeddings de los fragmentos que todavía no están en el
índice, y borra los de los archivos que cambiaron o que ya no existen.
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import ValidationError

from config import Settings
from documents import build_chunks
from embeddings import build_embeddings, embedding_dimension
from setup_index import IndexDimensionError, ensure_index

logger = logging.getLogger(__name__)

DELETE_BATCH = 500


@dataclass
class IngestReport:
    chunks: int = 0
    added: int = 0
    unchanged: int = 0
    deleted: int = 0


def source_of(vector_id: str) -> str:
    return vector_id.split("#", 1)[0]


def list_ids(index: Any, namespace: str) -> list[str]:
    return [id_ for page in index.list(namespace=namespace) for id_ in page]


def ingest(index: Any, vector_store: Any, chunks: list[Document], namespace: str) -> IngestReport:
    """Sincroniza el namespace del índice con los fragmentos calculados desde la carpeta."""
    report = IngestReport(chunks=len(chunks))
    wanted = {chunk.id for chunk in chunks}
    existing = set(list_ids(index, namespace))

    # Fragmentos viejos: de archivos modificados (cambió el hash) o que ya no están en la carpeta.
    stale = sorted(existing - wanted)
    for start in range(0, len(stale), DELETE_BATCH):
        index.delete(ids=stale[start : start + DELETE_BATCH], namespace=namespace)
    report.deleted = len(stale)

    new = [chunk for chunk in chunks if chunk.id not in existing]
    report.unchanged = len(chunks) - len(new)
    if new:
        vector_store.add_documents(new, ids=[chunk.id for chunk in new])
        report.added = len(new)

    by_source: dict[str, int] = defaultdict(int)
    for chunk in new:
        by_source[source_of(chunk.id)] += 1
    for source, count in sorted(by_source.items()):
        logger.info("%s: %d fragmentos nuevos", source, count)
    return report


def run_ingest(settings: Settings) -> IngestReport:
    embeddings = build_embeddings(settings)
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    ensure_index(
        pc,
        settings.index_name,
        embedding_dimension(embeddings),
        cloud=settings.cloud,
        region=settings.region,
    )

    index = pc.Index(settings.index_name)
    # El texto de cada fragmento se guarda en la metadata (text_key), así que al recuperar
    # no hace falta consultar ninguna otra base de datos.
    store = PineconeVectorStore(
        index=index, embedding=embeddings, text_key="text", namespace=settings.namespace
    )
    chunks = build_chunks(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
    return ingest(index, store, chunks, settings.namespace)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
        sys.exit(1)

    try:
        report = run_ingest(settings)
    except IndexDimensionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(
        f"Fragmentos: {report.chunks} | nuevos: {report.added} | "
        f"sin cambios: {report.unchanged} | borrados: {report.deleted}"
    )


if __name__ == "__main__":
    main()
