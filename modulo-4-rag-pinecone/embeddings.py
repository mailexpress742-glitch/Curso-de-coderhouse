from langchain_core.embeddings import Embeddings

from config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Embeddings locales (ONNX) con fastembed. El modelo se descarga la primera vez."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    return FastEmbedEmbeddings(settings.local_embedding_model)


def embedding_dimension(embeddings: Embeddings) -> int:
    """Dimensión de los vectores: el índice de Pinecone tiene que coincidir."""
    return len(embeddings.embed_query("dimensión"))
