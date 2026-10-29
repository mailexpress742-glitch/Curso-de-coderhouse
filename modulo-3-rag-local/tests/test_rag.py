import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from fakes import fake_chain, fake_embedder
from ingest import build_splitter, ingest
from rag import NOT_FOUND, RAGPipeline, build_answer_chain
from schemas import EmbeddingMismatchError, RAGError, RespuestaLLM
from store import open_collection

DOCS = {
    "redes.md": "El rate limiting restringe el ancho de banda y el tráfico de red por usuario.",
    "frutas.md": "Las manzanas y las peras son frutas que se cosechan en otoño.",
    "bases.md": "ChromaDB guarda vectores en disco con un cliente persistente.",
}


@pytest.fixture
def collection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name, text in DOCS.items():
        (data / name).write_text(text, encoding="utf-8")
    collection = open_collection(tmp_path / "store", fake_embedder())
    ingest(collection, data, build_splitter(500, 50))
    return collection


def pipeline(collection, answer: RespuestaLLM, top_k: int = 2) -> RAGPipeline:
    return RAGPipeline(collection, fake_chain(answer), top_k=top_k)


async def test_retrieval_returns_the_closest_chunk_first(collection):
    rag = pipeline(collection, RespuestaLLM(respuesta="x", encontrada=False))
    chunks = await rag.retrieve("limitar el tráfico de red y el ancho de banda")
    assert chunks[0].source == "redes.md"
    assert len(chunks) == 2


async def test_found_answer_lists_sources_from_retrieved_chunks(collection):
    answer = RespuestaLLM(respuesta="Restringe el tráfico.", encontrada=True, fragmentos_usados=[1])
    rag = pipeline(collection, answer)
    response = await rag.get_rag_response("¿Qué hace el rate limiting con el tráfico de red?")
    assert response.encontrada
    assert [s.archivo for s in response.fuentes] == ["redes.md"]
    assert response.fuentes[0].fragmento == 0


async def test_not_found_has_no_sources(collection):
    answer = RespuestaLLM(respuesta="No lo sé", encontrada=False)
    rag = pipeline(collection, answer)
    response = await rag.get_rag_response("¿Cuál es la capital de Australia?")
    assert not response.encontrada
    assert response.respuesta == NOT_FOUND
    assert response.fuentes == []


async def test_citing_a_chunk_that_does_not_exist_discards_the_answer(collection):
    answer = RespuestaLLM(respuesta="Inventado", encontrada=True, fragmentos_usados=[9])
    rag = pipeline(collection, answer)
    response = await rag.get_rag_response("tráfico de red")
    assert not response.encontrada
    assert response.fuentes == []


async def test_empty_question_is_rejected(collection):
    rag = pipeline(collection, RespuestaLLM(respuesta="x", encontrada=False))
    with pytest.raises(ValueError):
        await rag.get_rag_response("   ")


def test_found_answer_without_fragments_is_invalid():
    with pytest.raises(ValidationError):
        RespuestaLLM(respuesta="algo", encontrada=True, fragmentos_usados=[])


def test_opening_the_store_with_another_embedding_model_fails(tmp_path, collection):
    store = tmp_path / "store"
    with pytest.raises(EmbeddingMismatchError):
        open_collection(store, fake_embedder("fake:otro-modelo"))


def answer_chain_with(*replies: str):
    model = GenericFakeChatModel(messages=iter(AIMessage(content=text) for text in replies))
    return build_answer_chain(model, jitter=False)


GOOD_JSON = '{"respuesta": "Restringe el tráfico.", "encontrada": true, "fragmentos_usados": [1]}'


async def test_parser_turns_model_text_into_the_schema(collection):
    chain = answer_chain_with(f"```json\n{GOOD_JSON}\n```")
    rag = RAGPipeline(collection, chain, top_k=2)
    response = await rag.get_rag_response("tráfico de red y ancho de banda")
    assert response.encontrada
    assert response.fuentes[0].archivo == "redes.md"


async def test_malformed_model_output_is_retried(collection):
    chain = answer_chain_with("esto no es json", GOOD_JSON)
    rag = RAGPipeline(collection, chain, top_k=2)
    response = await rag.get_rag_response("tráfico de red y ancho de banda")
    assert response.encontrada


async def test_gives_up_when_the_model_never_returns_valid_json(collection):
    chain = answer_chain_with("mal", "mal", "mal")
    rag = RAGPipeline(collection, chain, top_k=2)
    with pytest.raises(RAGError):
        await rag.get_rag_response("tráfico de red")
