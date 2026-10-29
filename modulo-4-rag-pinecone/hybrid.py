import re
import unicodedata
from collections import Counter
from typing import Self

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import Settings
from documents import build_chunks
from embeddings import build_embeddings

STOPWORDS = {
    "a", "al", "con", "como", "cómo", "de", "del", "el", "en", "es", "la", "las", "lo", "los",
    "me", "para", "por", "que", "qué", "se", "su", "un", "una", "y",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes y sin palabras vacías. Conserva términos como model_validate."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return [word for word in re.findall(r"\w+", text) if word not in STOPWORDS]


class RAGSystem:
    """Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con un EnsembleRetriever.

    BM25 acierta con términos exactos y nombres propios (bump-pydantic, SecretStr), donde
    los embeddings pueden fallar. La búsqueda vectorial acierta con paráfrasis. El
    EnsembleRetriever combina las dos listas con Reciprocal Rank Fusion ponderado.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        chunks: list[Document],
        *,
        k: int = 5,
        bm25_weight: float = 0.4,
    ) -> None:
        self.k = k
        self.chunks = chunks
        self.vector_retriever = vector_retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=k, preprocess_func=tokenize)
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[bm25_weight, 1 - bm25_weight],
            id_key="chunk_id",
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        store = PineconeVectorStore(
            index=pc.Index(settings.index_name),
            embedding=build_embeddings(settings),
            text_key="text",
            namespace=settings.namespace,
        )
        vector_retriever = store.as_retriever(search_kwargs={"k": settings.top_k})
        # BM25 vive en memoria: se arma con los mismos fragmentos que se subieron a Pinecone.
        chunks = build_chunks(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
        return cls(vector_retriever, chunks, k=settings.top_k, bm25_weight=settings.bm25_weight)

    async def retrieve(self, query: str) -> list[Document]:
        """Los top-k fragmentos combinando resultados léxicos y semánticos."""
        docs = await self.ensemble.ainvoke(query)
        return docs[: self.k]

    async def retrieve_vector(self, query: str) -> list[Document]:
        return (await self.vector_retriever.ainvoke(query))[: self.k]

    async def retrieve_bm25(self, query: str) -> list[Document]:
        return (await self.bm25_retriever.ainvoke(query))[: self.k]

    def chunks_per_source(self) -> Counter[str]:
        return Counter(chunk.metadata["source"] for chunk in self.chunks)
