# Arquitectura RAG y respuestas fundamentadas

## Qué es RAG

RAG significa Retrieval-Augmented Generation, o generación aumentada por recuperación. En lugar de pedirle al modelo de lenguaje que responda solo con lo que aprendió durante su entrenamiento, el sistema primero busca información relevante en una base de conocimiento propia y se la entrega al modelo junto con la pregunta. Esto funciona como una memoria de largo plazo: el conocimiento se puede actualizar cambiando la base de datos, sin reentrenar el modelo.

## El flujo completo

El sistema tiene dos fases. La fase de ingesta se hace una vez, o cada vez que cambian los documentos: se leen los archivos, se limpian, se dividen en fragmentos, se calculan los embeddings y se guardan en la base vectorial.

La fase de consulta se hace en cada pregunta. Primero se convierte la pregunta en un embedding. Después se buscan los fragmentos más cercanos en la base vectorial. A continuación se arma un prompt que incluye esos fragmentos como contexto. Finalmente el modelo de lenguaje genera la respuesta usando ese contexto.

## Grounding

Un sistema RAG está fundamentado (grounded) cuando el modelo responde solo con la información recuperada. Para lograrlo, el prompt de sistema funciona como un filtro de veracidad: indica que debe responder únicamente con el contexto proporcionado y que, si la respuesta no está en el contexto, debe decir que no la sabe. Sin esa instrucción, el modelo tiende a completar con conocimiento propio, y puede inventar datos que parecen correctos pero no figuran en los documentos. Eso se llama alucinación.

## Cantidad de fragmentos

El parámetro `top_k` define cuántos fragmentos se recuperan. Un valor demasiado bajo puede dejar afuera la respuesta. Un valor demasiado alto agrega ruido, supera los límites de tokens y degrada la atención del modelo, un efecto conocido como Lost in the Middle, en el que el modelo presta menos atención a la información que está en el medio de un contexto largo. En la práctica se usa un `top_k` de entre 3 y 5.

## Trazabilidad de las fuentes

Una buena respuesta RAG no solo dice algo: indica de dónde lo sacó. Para eso se guardan metadatos con el archivo de origen de cada fragmento y se devuelven junto con la respuesta. Validar la salida con un modelo de Pydantic garantiza que la respuesta siempre tenga la misma estructura, con el texto y la lista de fuentes. Las fuentes deben construirse a partir de los fragmentos que realmente se recuperaron, y no a partir de lo que el modelo diga, para evitar referencias inventadas.

## Cómo probar un sistema RAG

Hacen falta como mínimo dos pruebas. La primera es una pregunta cuya respuesta sí está en los documentos: el sistema debe responder correctamente y citar la fuente. La segunda es una pregunta trampa, cuya respuesta no figura en los documentos: el sistema debe contestar que no lo sabe en lugar de inventar. Si responde con datos que no están en el contexto, el prompt de grounding no está funcionando.

## Ejecución asíncrona

Las llamadas a la base vectorial y al modelo de lenguaje son operaciones de entrada y salida que tardan. En una aplicación web conviene ejecutarlas de forma asíncrona para no bloquear el servidor mientras se espera la respuesta. Las cadenas de LangChain exponen `ainvoke` para esto, y las operaciones bloqueantes de una librería sincrónica se pueden envolver con `asyncio.to_thread`.
