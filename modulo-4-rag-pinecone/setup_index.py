"""Verifica que el índice de Pinecone exista y lo crea (Serverless) si hace falta.

python setup_index.py
"""

import logging
import sys
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pydantic import ValidationError

from config import Settings
from embeddings import build_embeddings, embedding_dimension

logger = logging.getLogger(__name__)


class IndexDimensionError(Exception):
    """El índice existente tiene otra dimensión que los embeddings configurados."""


def ensure_index(pc: Any, name: str, dimension: int, *, cloud: str, region: str) -> bool:
    """Devuelve True si creó el índice y False si ya existía."""
    if pc.has_index(name):
        existing = pc.describe_index(name).dimension
        if existing != dimension:
            raise IndexDimensionError(
                f"el índice '{name}' tiene dimensión {existing} pero los embeddings generan "
                f"vectores de {dimension}. Usá otro INDEX_NAME o borrá el índice desde Pinecone."
            )
        logger.info("el índice '%s' ya existe (dimensión %d)", name, existing)
        return False

    logger.info(
        "creando el índice serverless '%s' (%s/%s, dimensión %d)", name, cloud, region, dimension
    )
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    return True


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
        sys.exit(1)

    dimension = embedding_dimension(build_embeddings(settings))
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    try:
        created = ensure_index(
            pc, settings.index_name, dimension, cloud=settings.cloud, region=settings.region
        )
    except IndexDimensionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Índice '{settings.index_name}': {'creado' if created else 'ya existía'}")


if __name__ == "__main__":
    main()
