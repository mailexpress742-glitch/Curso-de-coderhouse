from fakes import FixedRetriever, make_chunk
from hybrid import RAGSystem, tokenize

CHUNKS = [
    make_chunk("migracion.md", "La herramienta bump-pydantic reescribe el código fuente."),
    make_chunk("config.md", "SecretStr oculta las claves de API en los logs."),
    make_chunk("validacion.md", "Field agrega restricciones numéricas como ge y le."),
    make_chunk("alias.md", "Un alias cambia el nombre del campo en los datos externos."),
    make_chunk("frutas.md", "Las manzanas y las peras se cosechan en otoño."),
    make_chunk("redes.md", "El rate limiting restringe el tráfico de red."),
]


def system(vector_docs, *, k=3, bm25_weight=0.5) -> RAGSystem:
    return RAGSystem(FixedRetriever(docs=vector_docs), CHUNKS, k=k, bm25_weight=bm25_weight)


def sources(docs) -> list[str]:
    return [doc.metadata["source"] for doc in docs]


def test_tokenize_drops_accents_stopwords_and_keeps_identifiers():
    assert tokenize("¿Cómo valido el campo model_validate?") == [
        "valido",
        "campo",
        "model_validate",
    ]


async def test_bm25_finds_an_exact_term_that_the_vector_search_missed():
    # La búsqueda vectorial devuelve documentos que no tienen nada que ver.
    rag = system([CHUNKS[4], CHUNKS[5], CHUNKS[3]])
    docs = await rag.retrieve("bump-pydantic")
    assert "migracion.md" in sources(docs)


async def test_vector_results_are_kept_in_the_combination():
    rag = system([CHUNKS[1], CHUNKS[3], CHUNKS[2]])
    docs = await rag.retrieve("bump-pydantic")
    assert {"config.md", "alias.md"} & set(sources(docs))


async def test_a_document_found_by_both_retrievers_comes_first_and_is_not_repeated():
    rag = system([CHUNKS[0], CHUNKS[4], CHUNKS[5]])
    docs = await rag.retrieve("bump-pydantic")
    assert sources(docs)[0] == "migracion.md"
    assert len(set(sources(docs))) == len(docs)


async def test_retrieve_returns_at_most_k_documents():
    rag = system([CHUNKS[1], CHUNKS[2], CHUNKS[3]], k=2)
    docs = await rag.retrieve("bump-pydantic")
    assert len(docs) <= 2


async def test_each_retriever_can_be_used_on_its_own():
    rag = system([CHUNKS[4]])
    assert sources(await rag.retrieve_vector("lo que sea")) == ["frutas.md"]
    assert sources(await rag.retrieve_bm25("bump-pydantic"))[0] == "migracion.md"


def test_chunks_per_source_counts_fragments():
    rag = system([])
    assert rag.chunks_per_source()["config.md"] == 1
