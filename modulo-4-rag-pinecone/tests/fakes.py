from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

Ids = list[str]


def make_chunk(source: str, text: str, index: int = 0) -> Document:
    chunk_id = f"{source}#{index:03d}-fake"
    return Document(
        id=chunk_id,
        page_content=text,
        metadata={
            "source": source,
            "page": 1,
            "chunk": index,
            "chunk_id": chunk_id,
            "categoria": "prueba",
            "etiquetas": ["x"],
        },
    )


class FixedRetriever(BaseRetriever):
    """Retriever vectorial falso: siempre devuelve los mismos documentos, en ese orden."""

    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return list(self.docs)


class FakeIndex:
    """Imita las operaciones de un índice de Pinecone que usa la ingesta."""

    def __init__(self, ids: Ids | None = None) -> None:
        self.ids = set(ids or [])
        self.deleted: Ids = []

    def list(self, namespace: str) -> Any:
        yield sorted(self.ids)

    def delete(self, ids: Ids, namespace: str) -> None:
        self.ids -= set(ids)
        self.deleted.extend(ids)


class FakeStore:
    def __init__(self, index: FakeIndex) -> None:
        self.index = index
        self.added: list[str] = []

    def add_documents(self, docs: list[Document], ids: list[str]) -> None:
        self.added.extend(ids)
        self.index.ids |= set(ids)


class FakeDescription:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


class FakePinecone:
    def __init__(self, existing: dict[str, int] | None = None) -> None:
        self.indexes = dict(existing or {})
        self.created: list[tuple[str, int, str]] = []

    def has_index(self, name: str) -> bool:
        return name in self.indexes

    def describe_index(self, name: str) -> FakeDescription:
        return FakeDescription(self.indexes[name])

    def create_index(self, name: str, dimension: int, metric: str, spec: Any) -> None:
        self.indexes[name] = dimension
        self.created.append((name, dimension, metric))
