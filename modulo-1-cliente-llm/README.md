# Módulo 1 · Unified Async LLM Client

Pre-entrega del módulo 1 de **AI Engineering** (Coderhouse): una capa de abstracción asíncrona para hablar con **OpenAI** y **Anthropic** a través de una sola interfaz, con validación Pydantic, streaming de tokens y manejo de errores.

```python
async with AsyncLLMManager.from_env() as llm:          # el proveedor sale de LLM_PROVIDER
    respuesta = await llm.generate("¿Qué es la entropía?")
    print(respuesta.content)

    async for chunk in llm.stream("¿Qué es la entropía?"):
        print(chunk.delta, end="", flush=True)
```

Para cambiar de OpenAI a Anthropic sólo hay que tocar `LLM_PROVIDER` en el `.env`. El código que usa el manager no cambia.

## Cómo ejecutarlo

Requiere **Python 3.12**.

1. Crear y activar el entorno virtual, e instalar dependencias:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Linux / macOS
   pip install -r requirements.txt
   ```

2. Copiar `.env.example` como `.env` y completar la key del proveedor que vas a usar.

3. Correr el script de prueba. Hace la pregunta "¿Qué es la entropía?" en modo normal y en streaming, y muestra un ejemplo de validación:

   ```bash
   python main.py                        # usa LLM_PROVIDER
   python main.py --provider anthropic   # fuerza un proveedor
   ```

4. Correr los tests. No necesitan API keys ni conexión, porque simulan las respuestas HTTP de las APIs:

   ```bash
   pytest
   ```

### Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `LLM_PROVIDER` | No (default `openai`) | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | Key de OpenAI |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | Key de Anthropic |
| `OPENAI_MODEL` | No (default `gpt-5-mini`) | Modelo de OpenAI |
| `ANTHROPIC_MODEL` | No (default `claude-opus-5-5`) | Modelo de Anthropic |
| `LLM_MAX_CONCURRENCY` | No (default `5`) | Máximo de llamadas simultáneas por manager |

## Estructura

```
modulo-1-cliente-llm/
├── schemas.py              # Pydantic: ChatMessage, GenerationParams, ModelResponse, StreamChunk, LLMError
├── config.py               # Settings: lee el entorno; las keys se guardan como SecretStr
├── clients/
│   ├── base.py             # BaseLLMClient (ABC): timeouts, reintentos, errores controlados
│   ├── openai_client.py    # OpenAIClient sobre AsyncOpenAI
│   └── anthropic_client.py # AnthropicClient sobre AsyncAnthropic
├── manager.py              # AsyncLLMManager + factory según configuración
├── main.py                 # script de validación (normal + streaming)
├── tests/test_clients.py   # tests con los SDKs reales y HTTP simulado
├── .env.example
└── requirements.txt
```

## Decisiones de diseño

**Abstracción y factory.** `BaseLLMClient` define la interfaz común: `generate()` y `stream()`. Cada proveedor implementa sólo tres piezas: cómo se llama a su SDK (`_generate`, `_stream`) y cómo se traducen sus excepciones (`_classify_error`). `AsyncLLMManager` elige la clase concreta según `LLM_PROVIDER` (patrón Factory) y es lo único que conoce el resto del código. Las diferencias entre SDKs quedan encapsuladas en cada cliente:

- OpenAI devuelve el texto en `choices[0].message.content`. Anthropic devuelve una lista de bloques (`content[i].text`), donde además puede haber bloques de razonamiento que se descartan.
- OpenAI recibe el prompt de sistema como un mensaje más. Anthropic lo recibe aparte, en el parámetro `system`.

**Asincronía.** Todas las llamadas usan los clientes asíncronos oficiales (`AsyncOpenAI`, `AsyncAnthropic`) con `await`, así que el event loop nunca se bloquea. Cada llamada a `generate()` está envuelta en `asyncio.timeout()`. Además, un `asyncio.Semaphore` en el manager limita cuántas llamadas corren a la vez, para no provocar errores 429 cuando se lanzan muchas tareas con `gather`.

**Streaming.** `stream()` es un generador asíncrono: recorre el stream del SDK con `async for` y hace `yield` de cada fragmento apenas llega. Cada fragmento es un `StreamChunk` con el texto nuevo en `delta`. El último chunk trae `done=True` junto con el motivo de fin, el uso de tokens o el error.

**Validación con Pydantic.**
- `GenerationParams` valida `temperature` (de 0 a 2), `max_tokens`, el timeout y la cantidad de reintentos, y rechaza parámetros desconocidos.
- `ChatRequest` valida que la conversación tenga al menos un mensaje y que el último sea del usuario.
- Las API keys se guardan como `SecretStr`, así que no aparecen en logs ni en tracebacks.

Las entradas inválidas lanzan `ValidationError` *antes* de llamar a la API: son errores de quien llama y conviene que fallen rápido.

**Resiliencia.** Los fallos de red, cuota o autenticación nunca lanzan excepciones. Vuelven como un `LLMError` estructurado, con tipo, mensaje, si se puede reintentar y código HTTP:

| Situación | Tipo | ¿Se reintenta? |
|---|---|---|
| Rate limit (429), sobrecarga (529), errores 5xx | `rate_limit` / `server` | Sí |
| Error de conexión o timeout | `connection` / `timeout` | Sí |
| Sin saldo en OpenAI (`insufficient_quota`) | `rate_limit` | No: reintentar no lo arregla |
| API key inválida (401/403) | `authentication` | No |
| Request inválido o modelo inexistente (400/404) | `invalid_request` | No |
| El modelo declina responder | `refusal` | No |

Los reintentos usan backoff exponencial con jitter, y si la API manda el header `retry-after`, se respeta. Los SDKs se crean con `max_retries=0` para que los reintentos no se multipliquen con los propios del SDK.

En streaming, sólo se reintenta si el fallo ocurre antes del primer token. Si ya se mostró texto, reintentar duplicaría la respuesta, así que el stream termina con un chunk de error.

**Detalles de cada proveedor.**
- `temperature` es opcional y sólo se envía si se pide. Los modelos de razonamiento de OpenAI rechazan valores distintos al default.
- Los modelos Claude actuales no aceptan `temperature`, así que `AnthropicClient` no lo envía y deja un aviso en el log.
- Para OpenAI se usa `max_completion_tokens`, que reemplaza a `max_tokens` y es el que aceptan los modelos de razonamiento.
