import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".md", ".txt"}
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​﻿\x0c]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Separa el bloque `clave: valor` entre `---` del principio del archivo."""
    raw = raw.replace("\r\n", "\n")
    match = FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip() and value.strip():
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Divide por párrafos, después por líneas y por espacios, midiendo en tokens."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )


def chunk_id(source: str, index: int, text: str) -> str:
    # El hash del contenido hace que el ID sea siempre el mismo para el mismo texto.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}#{index:03d}-{digest}"


def locate(body: str, piece: str, after: int) -> int:
    """Posición en el texto donde empieza un fragmento (-1 si no se encuentra)."""
    return body.find(piece[:80], after)


def section_number(body: str, start: int) -> int:
    """Número de sección (encabezado `## `) en la que empieza un fragmento."""
    return max(1, len(re.findall(r"^## ", body[: start + 3], flags=re.MULTILINE)))


def build_chunks(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Lee los archivos de la carpeta y los convierte en fragmentos con metadatos.

    Cada fragmento guarda en su metadata: la fuente, la página (sección del documento),
    la categoría y las etiquetas. El texto se guarda aparte, en el índice de Pinecone.
    """
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no hay archivos .md ni .txt en {data_dir}")

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for path in paths:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        body = clean_text(body)
        tags = [tag.strip() for tag in meta.get("etiquetas", "").split(",") if tag.strip()]

        start = 0
        for index, piece in enumerate(splitter.create_documents([body])):
            # Los fragmentos salen en orden, así que se busca a partir del anterior.
            start = max(locate(body, piece.page_content, start), start)
            id_ = chunk_id(path.name, index, piece.page_content)
            piece.id = id_
            piece.metadata = {
                "source": path.name,
                "page": section_number(body, start),
                "chunk": index,
                "chunk_id": id_,
                "categoria": meta.get("categoria", "sin-categoria"),
                "etiquetas": tags,
            }
            chunks.append(piece)
    return chunks
