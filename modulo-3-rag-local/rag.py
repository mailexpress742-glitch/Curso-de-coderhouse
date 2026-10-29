import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Self

from chromadb.api.models.Collection import Collection
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from pydantic import ValidationError

from config import Settings
from schemas import Fuente, Provider, RAGError, RespuestaLLM, RespuestaRAG
from store import build_embedder, open_collection

logger = logging.getLogger(__name__)

NOT_FOUND = "No lo sé: la respuesta no está en los documentos."

PARSER = PydanticOutputParser(pydantic_object=RespuestaLLM)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico. Respondé solo con la información del CONTEXTO. "
            "No uses conocimiento propio ni completes con suposiciones.\n"
            "- Si la respuesta está en el contexto, poné encontrada=true, respondé en español "
            "y listá en fragmentos_usados los números de los fragmentos en los que te apoyaste.\n"
            "- Si la respuesta no está en el contexto, poné encontrada=false, "
            'respondé "No lo sé" y dejá fragmentos_usados vacío.\n\n'
            "{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{context}\n\nPREGUNTA: {question}"),
    ]
).partial(format_instructions=PARSER.get_format_instructions())


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str
    distance: float

    def to_source(self) -> Fuente:
        extract = " ".join(self.text.split())
        if len(extract) > 200:
            extract = extract[:200].rstrip() + "…"
        return Fuente(
            archivo=self.source,
            fragmento=self.index,
            extracto=extract,
            distancia=round(self.distance, 4),
        )


def build_model(settings: Settings) -> BaseChatModel:
    key = settings.api_key.get_secret_value()
    if settings.provider == Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.model, api_key=key, max_tokens=2_000)


def build_answer_chain(
    model: BaseChatModel, *, max_attempts: int = 3, jitter: bool = True
) -> Runnable:
    """prompt | modelo | PydanticOutputParser. Reintenta si la salida no cumple el esquema."""
    return (PROMPT | model | PARSER).with_retry(
        retry_if_exception_type=(OutputParserException, ValidationError),
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=jitter,
    )


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{n}] (archivo: {chunk.source})\n{chunk.text}" for n, chunk in enumerate(chunks, start=1)
    )


class RAGPipeline:
    """Cadena LCEL completa: pregunta -> retriever -> contexto -> prompt -> modelo -> parser."""

    def __init__(self, collection: Collection, answer_chain: Runnable, *, top_k: int = 4) -> None:
        self.collection = collection
        self.top_k = top_k
        self.answer_chain = answer_chain
        self.chain = self._build_chain()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        collection = open_collection(settings.store_dir, build_embedder(settings))
        return cls(collection, build_answer_chain(build_model(settings)), top_k=settings.top_k)

    def _build_chain(self) -> Runnable:
        retriever = RunnableLambda(self.retrieve)

        async def answer(inputs: dict[str, Any]) -> RespuestaLLM:
            if not inputs["chunks"]:
                return RespuestaLLM(respuesta=NOT_FOUND, encontrada=False)
            return await self.answer_chain.ainvoke(
                {"context": inputs["context"], "question": inputs["question"]}
            )

        return (
            RunnableParallel(question=RunnablePassthrough(), chunks=retriever)
            | RunnablePassthrough.assign(context=lambda x: format_context(x["chunks"]))
            | RunnablePassthrough.assign(answer=RunnableLambda(answer))
        )

    async def retrieve(self, question: str) -> list[Chunk]:
        """Busca los fragmentos más parecidos a la pregunta."""
        # chromadb es sincrónico: va a un hilo para no bloquear el event loop.
        result: dict[str, Any] = await asyncio.to_thread(
            self.collection.query,
            query_texts=[question],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = [
            Chunk(meta["source"], meta["chunk"], text, distance)
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
        logger.info(
            "recuperados %d fragmentos: %s",
            len(chunks),
            [(c.source, c.index, round(c.distance, 3)) for c in chunks],
        )
        return chunks

    async def get_rag_response(self, query: str) -> RespuestaRAG:
        question = query.strip()
        if not question:
            raise ValueError("la pregunta está vacía")

        try:
            result = await self.chain.ainvoke(question)
        except (OutputParserException, ValidationError) as exc:
            logger.error("se agotaron los reintentos: %s", exc)
            raise RAGError("no se obtuvo una respuesta válida del modelo") from exc

        chunks: list[Chunk] = result["chunks"]
        answer: RespuestaLLM = result["answer"]

        # Las fuentes salen de los fragmentos recuperados, nunca del texto del modelo.
        used = sorted({n for n in answer.fragmentos_usados if 1 <= n <= len(chunks)})
        if not answer.encontrada or not used:
            if answer.encontrada:
                logger.warning("el modelo citó fragmentos que no existen: se descarta la respuesta")
            return RespuestaRAG(pregunta=question, respuesta=NOT_FOUND, encontrada=False)

        return RespuestaRAG(
            pregunta=question,
            respuesta=answer.respuesta,
            encontrada=True,
            fuentes=[chunks[n - 1].to_source() for n in used],
        )


@lru_cache(maxsize=1)
def default_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings(Settings.from_env())


async def get_rag_response(query: str, pipeline: RAGPipeline | None = None) -> RespuestaRAG:
    """Responde una pregunta usando solo los documentos indexados."""
    return await (pipeline or default_pipeline()).get_rag_response(query)
