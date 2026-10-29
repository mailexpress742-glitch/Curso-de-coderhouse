# Embeddings y similitud

## Qué es un embedding

Un embedding es una lista de números que representa el significado de un texto. Un modelo de embeddings recibe una frase o un párrafo y devuelve un vector de cientos o miles de dimensiones. Cuanto más parecido es el significado de dos textos, más cerca quedan sus vectores en ese espacio.

Esto resuelve el problema de la búsqueda por palabras clave. Si un documento habla de "restricción de ancho de banda" y el usuario busca "limitar el tráfico de red", una búsqueda por palabras exactas no lo encuentra. Con embeddings, los dos textos quedan cerca porque tratan del mismo tema, aunque no compartan ninguna palabra.

Antes de los embeddings densos se usaba la codificación one-hot, donde cada palabra era una posición en un vector enorme lleno de ceros. En ese esquema, la distancia entre "perro" y "canino" era igual a la distancia entre "perro" y "microprocesador", porque el vector no tenía ninguna noción de significado.

## Métricas de similitud

Hay tres métricas habituales para comparar vectores.

La similitud coseno mide el ángulo entre dos vectores. Captura la dirección y no la longitud, así que un párrafo largo y una oración corta sobre el mismo tema pueden resultar muy similares. Es la métrica más usada en sistemas de recuperación (RAG). En ChromaDB se configura con `hnsw:space` igual a `cosine`.

La distancia euclídea (L2) mide la distancia en línea recta entre dos puntos. Es sensible a la magnitud del vector, por lo que se usa sobre todo cuando los vectores están normalizados.

El producto punto multiplica las magnitudes y el coseno del ángulo. Se usa en sistemas de recomendación, donde la magnitud del vector expresa importancia.

Cuando una base de datos devuelve una distancia coseno, un valor cercano a 0 significa que los textos son muy parecidos, y un valor cercano a 1 significa que casi no se relacionan.

## Embeddings contextuales

Los modelos estáticos como Word2Vec o GloVe le asignan un único vector a cada palabra. La palabra "banco" tenía el mismo vector si se refería a un asiento o a una entidad financiera. Los modelos contextuales actuales generan un vector distinto según las palabras que rodean a cada término, así que distinguen los dos sentidos.

Un ejemplo de modelo moderno es `text-embedding-3-small` de OpenAI, que devuelve 1536 números por texto. Otro ejemplo es `all-MiniLM-L6-v2`, un modelo pequeño que se puede ejecutar localmente y devuelve 384 dimensiones.

## Errores comunes

El primer error es mezclar modelos. Un embedding generado con un modelo no se puede comparar con uno generado con otro modelo, porque viven en espacios vectoriales distintos y suelen tener dimensiones diferentes. Si se indexan documentos con un modelo y se consulta con otro, los resultados son ruido aleatorio.

El segundo error es ignorar el límite de tokens. Cada modelo de embeddings tiene una ventana de contexto máxima. Si se intenta convertir un documento muy largo en un solo vector, el modelo trunca el texto o pierde precisión. Por eso los documentos se dividen en fragmentos antes de indexarlos.

El tercer error es creer que los embeddings entienden la lógica. Capturan correlaciones estadísticas y pueden confundir negaciones: "hay stock" y "no hay stock" pueden quedar cerca porque ambas frases tratan sobre disponibilidad.

El cuarto error es el cambio de dominio. Un modelo entrenado con texto general de internet puede funcionar peor con lenguaje médico muy especializado o con código fuente.

## Aplicaciones

La búsqueda semántica recupera documentos por significado. La deduplicación agrupa tickets de soporte que preguntan lo mismo con palabras diferentes. La clasificación sin entrenamiento calcula el embedding de cada categoría y asigna cada texto a la categoría más cercana. Todas estas aplicaciones se apoyan en la misma idea: convertir texto en vectores y medir distancias.
