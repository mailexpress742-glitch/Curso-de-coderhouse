# Persistencia local con ChromaDB

## Qué es ChromaDB

ChromaDB es una base de datos vectorial de código abierto pensada para ser simple. Su modo local persistente es ideal para desarrollo y para aplicaciones privadas: no requiere levantar un contenedor ni un servidor, y guarda los datos en archivos dentro de una carpeta que elige el desarrollador.

Los embeddings que solo viven en memoria se pierden cuando termina el programa. Para que el conocimiento de un sistema sobreviva a un reinicio y pueda actualizarse sin volver a indexar todo, hay que usar un cliente persistente. En Python se crea con `chromadb.PersistentClient(path="./vectorstore")`. Si no se indica una ruta de disco, los datos se pierden al cerrar el script.

## Colecciones

Los datos se organizan en colecciones. Una colección guarda tres cosas por cada elemento: el texto del documento, su vector y sus metadatos. La métrica de distancia se define al crear la colección, por ejemplo con `metadata={"hnsw:space": "cosine"}`. Una vez creada, esa métrica no se cambia.

## Operaciones CRUD

Create: se agregan elementos con `add` o `upsert`. No alcanza con guardar el texto, porque también hay que guardar su vector. Si el texto cambia, el vector se tiene que volver a calcular. La buena práctica es usar `upsert`: si el ID ya existe, actualiza el elemento; si no existe, lo crea.

Read: hay dos formas de leer. El método `get` busca por IDs exactos o por filtros de metadatos, como un SELECT de una base de datos tradicional. El método `query` busca por similitud semántica y es el corazón de un sistema RAG. Recibe el texto de la pregunta, lo convierte en un vector y devuelve los fragmentos más cercanos junto con su distancia.

Update: permite cambiar el documento o sus metadatos. Si cambia el documento, se dispara una nueva vectorización.

Delete: elimina los elementos que ya no son relevantes, por ID o por filtro de metadatos.

## Metadatos

Los metadatos son etiquetas que se guardan junto a cada vector, por ejemplo `{"source": "manual.md", "chunk": 3}`. Sirven para dos cosas. La primera es filtrar los resultados antes de la búsqueda vectorial. La segunda es rastrear el origen de cada fragmento, lo que permite citar las fuentes en la respuesta final.

## IDs deterministas

Un error común es usar IDs aleatorios. Con IDs aleatorios, volver a indexar el mismo documento crea duplicados y hace imposible actualizar de forma eficiente. Se recomienda derivar el ID del contenido, por ejemplo con un hash del texto, o armar IDs estructurados como el nombre del archivo más el número de fragmento. Así, indexar dos veces el mismo contenido no produce duplicados.

## Función de embeddings

La colección usa una función de embeddings para vectorizar tanto los documentos que se agregan como las preguntas que se consultan. Es fundamental usar el mismo modelo en ambos momentos. Si una base se creó con un modelo y se consulta con otro, los resultados no tienen sentido. Por eso conviene guardar el nombre del modelo en los metadatos de la colección y verificarlo cada vez que se abre.

## Evitar reindexar

Indexar tiene un costo de tiempo y, con modelos de pago, de dinero. Antes de indexar conviene comprobar qué fragmentos ya existen en la base y procesar solo los nuevos o los que cambiaron. Si un archivo fue modificado, hay que borrar sus fragmentos viejos para que no queden respuestas desactualizadas.
