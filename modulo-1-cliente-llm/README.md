# Cliente asíncrono unificado para LLMs

Pre-entrega del módulo 1 de AI Engineering (Coderhouse).

Es una capa que permite usar OpenAI o Anthropic con la misma interfaz. Todas las llamadas son asíncronas, los parámetros se validan con Pydantic, las respuestas se pueden recibir en streaming y los errores de red o de cuota no cortan el programa.

```python
async with AsyncLLMManager.from_env() as llm:
    respuesta = await llm.generate("¿Qué es la entropía?")
    print(respuesta.content)

    async for chunk in llm.stream("¿Qué es la entropía?"):
        print(chunk.delta, end="")
```

El proveedor se elige con la variable `LLM_PROVIDER`, así que pasar de uno a otro no requiere cambiar código.

## Instalación

Hace falta Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Después copiar `.env.example` como `.env` y completar la API key del proveedor que se vaya a usar.

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` | | Obligatoria si se usa OpenAI |
| `ANTHROPIC_API_KEY` | | Obligatoria si se usa Anthropic |
| `OPENAI_MODEL` | `gpt-5-mini` | |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | |
| `LLM_MAX_CONCURRENCY` | `5` | Máximo de llamadas en paralelo |

## Uso

`main.py` hace la pregunta "¿Qué es la entropía?" en modo normal y en streaming, y muestra un ejemplo de validación:

```bash
python main.py
python main.py --provider anthropic
```

Los tests no usan API keys ni red: los SDKs reales responden con respuestas HTTP simuladas.

```bash
pytest
```

## Estructura

```
schemas.py              modelos Pydantic: mensajes, parámetros, respuestas y errores
config.py               configuración desde variables de entorno
clients/
  base.py               BaseLLMClient: timeouts, reintentos y manejo de errores
  errors.py             conversión de errores HTTP a LLMError
  openai_client.py      OpenAIClient (AsyncOpenAI)
  anthropic_client.py   AnthropicClient (AsyncAnthropic)
manager.py              AsyncLLMManager y la factory que elige el cliente
main.py                 script de prueba
tests/
```

## Cómo funciona

- `BaseLLMClient` es una clase abstracta. Cada proveedor implementa cómo llamar a su SDK y cómo traducir sus excepciones; el resto (timeouts, reintentos, errores) está en la base y es igual para los dos.
- `AsyncLLMManager.from_env()` arma el cliente correcto según la configuración. También limita la cantidad de llamadas simultáneas con un `asyncio.Semaphore`.
- `stream()` es un generador asíncrono que devuelve cada fragmento de texto apenas llega. El último fragmento trae `done=True` con el motivo de fin y el uso de tokens.
- Las entradas inválidas, como `temperature=3`, lanzan `ValidationError` antes de llamar a la API.
- Los errores de la API no lanzan excepciones: vuelven en `response.error` con el tipo de error y si se puede reintentar.
- Los rate limits, los errores 5xx y los cortes de red se reintentan con backoff exponencial, respetando el header `retry-after`. Una API key inválida, un request mal armado o una cuenta sin saldo no se reintentan.
- En streaming solo se reintenta si el error ocurre antes del primer token, para no repetir texto que ya se mostró.

## Diferencias entre proveedores

- OpenAI devuelve el texto en `choices[0].message.content`. Anthropic devuelve una lista de bloques, donde además puede haber bloques de razonamiento que no se muestran.
- OpenAI recibe el prompt de sistema como un mensaje más. Anthropic lo recibe aparte, en el parámetro `system`.
- Los modelos actuales de Claude no aceptan `temperature`, así que el cliente de Anthropic no la envía y deja un aviso en el log. En OpenAI solo se envía si se define.
