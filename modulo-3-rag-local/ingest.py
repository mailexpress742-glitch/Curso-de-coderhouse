"""Lee los archivos de data/, los limpia, los fragmenta y los guarda en ChromaDB.

python ingest.py              # solo indexa lo que cambió
python ingest.py --rebuild    # borra la base y la arma de nuevo
"""

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

from config import Settings
from store import build_embedder, open_collection

logger = logging.getLogger(__name__)

EXTENSIONS = {".txt", ".md"}


@dataclass
class IngestReport:
    files: int = 0
    added: int = 0
    unchanged: int = 0
    deleted: int = 0


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​﻿\x0c]", "", text)  # caracteres invisibles y saltos de página
    text = re.sub(r"[ \t]+\n", "\n", text)  # espacios al final de la línea
    text = re.sub(r"\n{3,}", "\n\n", text)  # más de una línea en blanco seguida
    return text.strip()


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Divide por párrafos, después por líneas y por espacios, midiendo en tokens."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_id(source: str, index: int, text: str) -> str:
    # El hash del contenido hace que el ID sea siempre el mismo para el mismo texto.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}-{index:03d}-{digest}"


def list_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in EXTENSIONS)


def ingest(
    collection: Collection, data_dir: Path, splitter: RecursiveCharacterTextSplitter
) -> IngestReport:
    """Sincroniza la colección con la carpeta: agrega lo nuevo y borra lo que ya no está.

    Solo calcula embeddings de los fragmentos que todavía no existen en la base.
    """
    report = IngestReport()
    paths = list_files(data_dir)
    if not paths:
        raise FileNotFoundError(f"no hay archivos .txt ni .md en {data_dir}")

    for path in paths:
        report.files += 1
        chunks = splitter.split_text(clean_text(path.read_text(encoding="utf-8")))
        ids = [chunk_id(path.name, i, chunk) for i, chunk in enumerate(chunks)]

        current = set(collection.get(where={"source": path.name}, include=[])["ids"])
        stale = sorted(current - set(ids))
        if stale:
            collection.delete(ids=stale)
            report.deleted += len(stale)

        new = [
            (i, id_, chunk)
            for i, (id_, chunk) in enumerate(zip(ids, chunks, strict=True))
            if id_ not in current
        ]
        report.unchanged += len(ids) - len(new)
        if new:
            collection.upsert(
                ids=[id_ for _, id_, _ in new],
                documents=[chunk for _, _, chunk in new],
                metadatas=[{"source": path.name, "chunk": i} for i, _, _ in new],
            )
            report.added += len(new)
        logger.info("%s: %d fragmentos (%d nuevos)", path.name, len(ids), len(new))

    # Archivos que se borraron de la carpeta.
    known = {p.name for p in paths}
    for metadata in collection.get(include=["metadatas"])["metadatas"]:
        if metadata["source"] not in known:
            collection.delete(where={"source": metadata["source"]})
            report.deleted += 1
    return report


def run_ingest(settings: Settings, *, rebuild: bool = False) -> IngestReport:
    collection = open_collection(settings.store_dir, build_embedder(settings), rebuild=rebuild)
    splitter = build_splitter(settings.chunk_size, settings.chunk_overlap)
    return ingest(collection, settings.data_dir, splitter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de documentos en ChromaDB")
    parser.add_argument("--rebuild", action="store_true", help="borra la base y la arma de nuevo")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        sys.exit(1)

    report = run_ingest(settings, rebuild=args.rebuild)
    print(
        f"Archivos: {report.files} | fragmentos nuevos: {report.added} | "
        f"sin cambios: {report.unchanged} | borrados: {report.deleted}"
    )


if __name__ == "__main__":
    main()
