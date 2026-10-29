# Sistema RAG escalable en la nube con Pinecone

Pre-entrega del módulo 4 de AI Engineering (Coderhouse).

Recuperador de documentos que combina búsqueda semántica en un índice de Pinecone Serverless con búsqueda léxica BM25. Los documentos de ejemplo son la documentación de una librería de Python (Pydantic v2), escrita en Markdown. Incluye un script de evaluación que mide Precision@5 y Recall@5 con un golden set de 18 preguntas.

```python
rag = RAGSystem.from_settings(Settings.from_env())
docs = await rag.retrieve("¿Cómo evito que una clave de API aparezca en los logs?")
for doc in docs:
    print(doc.metadata["source"], doc.metadata["categoria"], doc.page_content[:80])
```

## Instalación

Hace falta Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo replicar el índice de Pinecone

1. Crear una cuenta en [Pinecone](https://app.pinecone.io) (alcanza el plan gratuito) y copiar una API key.
2. Copiar `.env.example` como `.env` y completar `PINECONE_API_KEY`. Con los valores por defecto el índice se llama `rag-pydantic-docs` y se crea en AWS `us-east-1`.
3. Crear el índice. El script verifica si existe y, si no, lo crea en modo Serverless con métrica coseno:

   ```bash
   python setup_index.py
   ```

4. Subir los documentos de `data/`:

   ```bash
   python ingest.py
   ```

5. Probar una consulta y correr la evaluación:

   ```bash
   python main.py "¿Cómo oculto una contraseña en la configuración?"
   python evaluate.py
   ```

`ingest.py` ya crea el índice si falta, así que el paso 3 es opcional. Se puede correr las veces que se quiera: solo sube los fragmentos que cambiaron.

### Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `PINECONE_API_KEY` | | Obligatoria |
| `INDEX_NAME` | `rag-pydantic-docs` | Nombre del índice |
| `PINECONE_NAMESPACE` | `pydantic-docs` | Namespace dentro del índice |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | Dónde se crea el índice |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (sin key) u `openai` |
| `LOCAL_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-es` | Modelo local (768 dimensiones) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de OpenAI (1536 dimensiones) |
| `OPENAI_API_KEY` | | Solo si `EMBEDDINGS_PROVIDER=openai` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `600` / `60` | Tokens por fragmento y solapamiento |
| `RAG_TOP_K` | `5` | Fragmentos que devuelve el recuperador |
| `BM25_WEIGHT` | `0.4` | Peso de BM25 (el de la búsqueda vectorial es el resto) |

La dimensión del índice tiene que coincidir con la de los embeddings: 768 con el modelo local y 1536 con `text-embedding-3-small`. Si el índice ya existe con otra dimensión, los scripts se detienen con un mensaje claro. En ese caso hay que usar otro `INDEX_NAME` o borrar el índice desde la consola de Pinecone. Con `EMBEDDINGS_PROVIDER=local`, el modelo (unos 640 MB) se descarga la primera vez que se usa.

## Estructura

```
data/            12 documentos Markdown sobre Pydantic v2, con categoría y etiquetas
golden_set.json  18 preguntas con el documento que contiene la respuesta
config.py        configuración desde variables de entorno
embeddings.py    embeddings locales (fastembed) o de OpenAI
documents.py     limpieza, chunking y metadatos
setup_index.py   verifica o crea el índice Serverless
ingest.py        sube los fragmentos a Pinecone
hybrid.py        RAGSystem: EnsembleRetriever (BM25 + Pinecone)
evaluate.py      Precision@k y Recall@k
main.py          consulta de ejemplo
tests/
```

## Cómo funciona

**Ingesta (`ingest.py`)**

1. Lee los `.md` de `data/`. Cada archivo empieza con un bloque con su `categoria` y sus `etiquetas`.
2. Los limpia y los divide con `RecursiveCharacterTextSplitter`, midiendo en tokens: fragmentos de 600 tokens con 60 de overlap, dentro del rango de 500 a 800 recomendado.
3. Cada fragmento lleva metadatos: `source` (archivo), `page` (número de la sección del documento donde empieza), `categoria`, `etiquetas`, `chunk` y `chunk_id`.
4. Los embeddings se calculan y se suben a Pinecone con `PineconeVectorStore`. El texto original de cada fragmento se guarda en la metadata (`text`), así que al recuperar no hace falta otra base de datos.
5. Los fragmentos van a un namespace propio (`PINECONE_NAMESPACE`), para no mezclar este conjunto con otros datos del mismo índice.
6. Los IDs salen del archivo, la posición y un hash del contenido. Antes de subir, el script lista los IDs del namespace: sube solo los nuevos y borra los de archivos que cambiaron o desaparecieron. Volver a correrlo no cuesta nada.

**Recuperador híbrido (`hybrid.py`)**

`RAGSystem` encapsula un `EnsembleRetriever` con dos recuperadores:

- `BM25Retriever` (léxico): puntúa por coincidencia exacta de términos. Funciona bien con nombres propios y términos técnicos como `bump-pydantic` o `SecretStr`. Vive en memoria y se arma con los mismos fragmentos que se subieron a Pinecone. La tokenización saca tildes y palabras vacías.
- Recuperador de Pinecone (semántico): encuentra fragmentos con el mismo significado aunque usen otras palabras.

El `EnsembleRetriever` combina las dos listas con Reciprocal Rank Fusion ponderado (`BM25_WEIGHT` para BM25 y el resto para Pinecone) y `retrieve()` devuelve los 5 mejores fragmentos.

**Evaluación (`evaluate.py`)**

`golden_set.json` tiene 18 pares `{"pregunta": ..., "documento_id_esperado": ...}`, donde el documento esperado es el archivo que contiene la respuesta. Las preguntas mezclan términos exactos (`bump-pydantic`, `extra`, `errors()`) con paráfrasis que no repiten las palabras del documento. Para cada pregunta se recuperan los 5 mejores fragmentos y se calcula:

- Recall@5: 1 si el documento esperado aparece entre los 5 recuperados y 0 si no. Se promedia sobre las preguntas.
- Precision@5: fracción de los 5 fragmentos recuperados que pertenecen al documento esperado.
- Precisión normalizada: la misma precisión, pero dividida por el máximo alcanzable. Cada documento tiene 2 fragmentos, así que nunca puede aportar más de 2 de los 5 resultados y la Precision@5 no pasa de 0.40.

El reporte compara el recuperador híbrido con la búsqueda vectorial sola y con BM25 solo, agrega el Recall@1 (el primer resultado es el documento correcto) y muestra el detalle por pregunta.

## Resultados

Salida de `python evaluate.py` contra un índice Serverless real de Pinecone (AWS `us-east-1`, 768 dimensiones, embeddings locales). Dieron los mismos números que una corrida con un almacén vectorial en memoria y los mismos embeddings, lo que confirma que la ingesta a Pinecone no pierde información.

```
=== Evaluación: 18 preguntas sobre 24 fragmentos (k=5) ===
recuperador   Recall@5   Precision@5  Prec.norm.  Recall@1
híbrido           0.89          0.27        0.67      0.61
vectorial         0.89          0.27        0.67      0.56
BM25              0.83          0.24        0.61      0.61
```

Cómo leerlos:

- Con 24 fragmentos y 5 resultados, el Recall@5 ya no es trivial: una búsqueda al azar acertaría en torno al 21 % de las preguntas, contra el 89 % del híbrido.
- El híbrido iguala a la búsqueda vectorial en Recall@5 y la supera en Recall@1 (0.61 contra 0.56). Supera a BM25 solo en Recall@5 (0.89 contra 0.83).
- Son 18 preguntas, así que una diferencia de una pregunta mueve las métricas unos 0.06. Estos números no alcanzan para afirmar que el híbrido sea mejor que la búsqueda vectorial sola.
- Las dos preguntas que fallan (sobre `discriminator` y sobre actualizaciones parciales) se contestan en un solo fragmento, y el recuperador trae otros fragmentos parecidos de otros documentos.

## Tests

Los tests no usan Pinecone ni red: reemplazan el índice por uno falso y el recuperador vectorial por uno con resultados fijos.

```bash
pytest
```
