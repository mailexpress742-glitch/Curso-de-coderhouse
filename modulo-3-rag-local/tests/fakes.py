import hashlib
import re
from typing import Any

from chromadb import EmbeddingFunction
from langchain_core.runnables import RunnableLambda

from schemas import RespuestaLLM
from store import Embedder

DIMENSIONS = 128


class BagOfWords(EmbeddingFunction):
    """Embeddings falsos y deterministas: cuenta palabras en un vector de tamaño fijo.

    Textos con palabras en común quedan cerca, lo que alcanza para probar la recuperación.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS
            vector[slot] += 1.0
        return vector

    @staticmethod
    def name() -> str:
        return "bag-of-words"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "BagOfWords":
        return BagOfWords()


def fake_embedder(name: str = "fake:bag-of-words") -> Embedder:
    return Embedder(BagOfWords(), name)


def fake_chain(answer: RespuestaLLM) -> RunnableLambda:
    """Reemplaza al modelo de lenguaje: devuelve siempre la misma respuesta."""
    return RunnableLambda(lambda _: answer)
