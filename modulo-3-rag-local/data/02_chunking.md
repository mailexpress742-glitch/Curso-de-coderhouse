# Estrategias de chunking

## Por qué hay que fragmentar los documentos

Los modelos de embeddings tienen una ventana de contexto limitada, y un vector que resume un documento entero pierde los detalles. Si se convierte un PDF de 50 páginas en un único vector, el resultado es un promedio semántico tan diluido que no sirve para encontrar respuestas específicas.

El objetivo del chunking es crear unidades de información pequeñas y con sentido propio. Cada fragmento (chunk) debe poder entenderse por sí solo y tratar de un tema acotado.

## Limpieza previa

Antes de dividir conviene limpiar el texto. Los documentos reales suelen traer encabezados y pies de página repetidos, saltos de línea dobles o triples, caracteres raros que quedan de una mala extracción de tablas y rutas de archivos. Si un pie de página queda en medio de una frase importante, el modelo de embeddings captura ese ruido y empeora la búsqueda. La regla es simple: si entra basura, sale basura.

## Estrategias

El tamaño fijo corta el texto cada cierta cantidad de caracteres o tokens. Es la estrategia más simple y también la más peligrosa, porque puede cortar una oración o un concepto por la mitad.

El Recursive Character Splitting es el estándar en LangChain. Intenta dividir con una jerarquía de separadores: primero párrafos (doble salto de línea), después saltos de línea simples, después espacios y, como último recurso, caracteres individuales. Así, en la medida de lo posible, los párrafos y las oraciones se mantienen juntos. En LangChain se implementa con la clase `RecursiveCharacterTextSplitter`.

## Overlap

El overlap o solapamiento hace que el final de un fragmento sea también el comienzo del siguiente. Sirve para el caso en que una idea importante queda justo en el límite entre dos fragmentos: con solapamiento, el contexto se conserva en ambos, y la recuperación funciona sin importar por dónde se haya cortado el texto.

Una configuración habitual es un tamaño de 500 tokens con un solapamiento de 50 tokens, es decir, alrededor del 10 por ciento.

## Medir en tokens, no en caracteres

Los modelos cobran y limitan por tokens, no por caracteres. Por eso conviene medir el tamaño de los fragmentos con un tokenizador real, como `tiktoken`. En LangChain, el método `from_tiktoken_encoder` del splitter permite expresar `chunk_size` y `chunk_overlap` directamente en tokens.

## Errores comunes

Los fragmentos demasiado pequeños generan ruido y pierden el contexto global: el modelo de lenguaje no logra entender de qué trata el fragmento. Los fragmentos demasiado grandes diluyen el significado y gastan tokens del contexto. No limpiar el texto reduce la precisión de la búsqueda. Y medir en caracteres en lugar de tokens puede producir fragmentos que superan los límites del modelo.

## Cómo elegir el tamaño

Un buen chunking es un equilibrio entre tres cosas. La granularidad pide fragmentos pequeños para que la búsqueda sea precisa. El contexto pide fragmentos lo bastante grandes, o con solapamiento, para que se entiendan. La limpieza pide datos normalizados para que los embeddings sean precisos. No existe un tamaño ideal para todos los casos: conviene probar con preguntas reales y revisar qué fragmentos se recuperan.
