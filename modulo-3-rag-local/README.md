# Sistema de recuperación semántica local (RAG)

Pre-entrega del módulo 3 de AI Engineering (Coderhouse).

Responde preguntas usando solo los documentos de la carpeta `data/`. Los documentos se limpian, se fragmentan y se guardan en una base ChromaDB local. Cuando llega una pregunta, el sistema busca los fragmentos más parecidos, se los pasa al modelo de lenguaje y devuelve una respuesta validada con Pydantic, con las fuentes de donde salió. Si la respuesta no está en los documentos, dice "No lo sé".

```python
respuesta = await get_rag_response("¿Para qué sirve el overlap al fragmentar documentos?")
print(respuesta.model_dump_json(indent=2))
```

```json
{
  "pregunta": "¿Para qué sirve el overlap al fragmentar documentos?",
  "respuesta": "El overlap hace que el final de un fragmento sea también el comienzo del siguiente...",
  "encontrada": true,
  "fuentes": [
    {
      "archivo": "02_chunking.md",
      "fragmento": 0,
      "extracto": "# Estrategias de chunking ...",
      "distancia": 0.3747
    }
  ]
}
```

## Instalación

Hace falta Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Después copiar `.env.example` como `.env` y completar la API key del proveedor del modelo de lenguaje.

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | | Obligatoria si se usa OpenAI (modelo o embeddings) |
| `ANTHROPIC_API_KEY` | | Obligatoria si se usa Anthropic |
| `OPENAI_MODEL` | `gpt-5-mini` | |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (sin key) u `openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Solo con embeddings de OpenAI |
| `RAG_CHUNK_SIZE` | `500` | Tokens por fragmento |
| `RAG_CHUNK_OVERLAP` | `50` | Tokens de solapamiento |
| `RAG_TOP_K` | `5` | Fragmentos que se recuperan por pregunta |

Con `EMBEDDINGS_PROVIDER=local` los embeddings los calcula el modelo `all-MiniLM-L6-v2` en tu máquina. La primera vez que se usa, ChromaDB lo descarga (unos 80 MB).

## Uso

Indexar los documentos. Se puede correr las veces que se quiera: solo procesa lo que cambió.

```bash
python ingest.py
python ingest.py --rebuild    # borra la base y la arma de nuevo
```

Probar el sistema con dos preguntas, una cuya respuesta está en los documentos y una "pregunta trampa" cuya respuesta no está. `main.py` indexa primero si hace falta.

```bash
python main.py
python main.py --provider anthropic
python main.py "¿Qué es un embedding?"    # una pregunta propia
```

Los tests no usan API keys, red ni el modelo de embeddings real: usan embeddings falsos y un modelo de lenguaje falso (incluyendo JSON mal formado, para probar los reintentos).

```bash
pytest
```

## Estructura

```
data/          4 documentos de ejemplo sobre embeddings, chunking, ChromaDB y RAG
schemas.py     modelos Pydantic: Fuente, RespuestaLLM, RespuestaRAG y errores
config.py      configuración desde variables de entorno
store.py       cliente persistente de ChromaDB y función de embeddings
ingest.py      limpieza, chunking e ingesta (script de carga)
rag.py         recuperación, cadena LCEL y get_rag_response()
main.py        script de prueba
tests/
vectorstore/   la base de ChromaDB (se crea sola, no se sube al repo)
```

## Cómo funciona

**Ingesta (`ingest.py`)**

1. Lee los `.txt` y `.md` de `data/` y limpia el texto: unifica los saltos de línea, saca caracteres invisibles y colapsa las líneas en blanco repetidas.
2. Los divide con `RecursiveCharacterTextSplitter`, que prueba separar por párrafos, después por líneas y por espacios. El tamaño se mide en tokens (`tiktoken`): fragmentos de 500 tokens con 50 de overlap.
3. Cada fragmento tiene un ID determinista (archivo, posición y hash del contenido) y guarda en sus metadatos el archivo y la posición.
4. Antes de guardar, compara con lo que ya hay en la base. Calcula embeddings solo de los fragmentos nuevos, borra los de los archivos que cambiaron o desaparecieron y no toca el resto. Por eso volver a correrlo no cuesta nada.

**Consulta (`rag.py`)**

Toda la consulta es una sola cadena LCEL:

```
pregunta -> retriever -> contexto -> prompt -> modelo -> PydanticOutputParser -> RespuestaLLM
```

1. El retriever (`retrieve()`) busca en ChromaDB los `top_k` fragmentos más cercanos a la pregunta (distancia coseno). Es el primer eslabón de la cadena y corre en un hilo con `asyncio.to_thread`, para no bloquear el event loop.
2. Los fragmentos se numeran y se arman como contexto. El prompt pide responder solo con ese contexto y decir "No lo sé" si la respuesta no está.
3. `PydanticOutputParser` convierte la respuesta del modelo en un `RespuestaLLM`: la respuesta, si la encontró y qué fragmentos usó. Si el modelo no devuelve un JSON válido, la cadena reintenta hasta 3 veces (`.with_retry()`).
4. `get_rag_response()` ejecuta la cadena con `ainvoke` y arma la `RespuestaRAG` final. Las fuentes salen de los fragmentos que realmente se recuperaron, no del texto del modelo, así que no puede citar archivos inventados. Si el modelo dice que encontró la respuesta pero cita fragmentos que no existen, la respuesta se descarta.

**Protecciones**

- La base guarda el nombre del modelo de embeddings con el que se creó. Si después se configura otro, el sistema se detiene con un mensaje claro en vez de devolver resultados sin sentido (hay que correr `python ingest.py --rebuild`).
- La API key nunca está en el código: se lee del `.env`, que está ignorado por git.

## Resultados de la prueba

Con `python main.py`:

- La pregunta cuya respuesta está en los documentos devuelve la respuesta y la fuente `02_chunking.md`.
- La pregunta trampa ("¿Cuánto cuesta por mes el plan de Pinecone...?") devuelve `encontrada: false` y "No lo sé", sin fuentes.

## Limitaciones

El modelo de embeddings local procesa unos 256 tokens por texto, y los fragmentos son de 500. Lo que queda al final de cada fragmento no influye en su vector, así que la búsqueda es menos precisa que con un modelo de ventana más grande. Con `EMBEDDINGS_PROVIDER=openai` el problema no existe, porque `text-embedding-3-small` acepta fragmentos de ese tamaño. Después de cambiar de modelo hay que reconstruir la base con `--rebuild`.
