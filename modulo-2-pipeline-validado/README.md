# Pipeline de extracción técnica validado

Pre-entrega del módulo 2 de AI Engineering (Coderhouse).

Recibe un texto sin procesar (una descripción de arquitectura, un log de error) y devuelve un objeto validado con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico. Está armado con LangChain (LCEL) y Pydantic, y reutiliza la selección de proveedor del [módulo 1](../modulo-1-cliente-llm/): OpenAI o Anthropic según una variable de entorno.

```python
chain = build_chain(build_model(Settings.from_env()))
resultado = await process_text("Nuestra API en FastAPI usa Redis y PostgreSQL...", chain)
print(resultado.model_dump_json(indent=2))
```

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "PostgreSQL alcanza su límite de conexiones en picos de carga y la API devuelve 500 en el checkout."
}
```

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
| `PIPELINE_MAX_ATTEMPTS` | `3` | Intentos máximos si la salida no cumple el esquema |

## Uso

`main.py` procesa dos textos (una descripción de arquitectura y un texto ambiguo para probar el pipeline) y muestra el resultado y los logs:

```bash
python main.py
python main.py --provider anthropic
```

Los tests no usan API keys ni red: reemplazan al modelo por uno falso que devuelve respuestas armadas.

```bash
pytest
```

## Estructura

```
schemas.py   ExtraccionTecnica (Pydantic), el enum NivelCriticidad y los errores del pipeline
chain.py     prompt, modelo, cadena LCEL con reintentos y process_text()
config.py    configuración desde variables de entorno
main.py      script de prueba
tests/
```

## Cómo funciona

La cadena es `prompt | modelo con salida estructurada | validación`, con reintentos:

1. `ChatPromptTemplate` arma los mensajes (sistema y usuario) a partir de la variable `{text}`.
2. `model.with_structured_output(ExtraccionTecnica, method="json_schema", include_raw=True)` le pide al modelo que responda con el esquema y lo convierte en un objeto Pydantic.
3. `check_output` revisa el resultado: si la respuesta quedó cortada (`finish_reason` de tipo `length` o `max_tokens`), si no cumple el esquema o si no llegó ningún objeto, lanza un error.
4. `.with_retry()` envuelve los pasos 2 y 3. Si hay un error de formato, vuelve a llamar al modelo, con espera exponencial, hasta `PIPELINE_MAX_ATTEMPTS` intentos. Si se agotan, `process_text()` lanza `PipelineError`.

Detalles a tener en cuenta:

- Los errores de red y los rate limits no se reintentan en la cadena: ya los reintentan los SDKs de OpenAI y Anthropic. Así no se multiplican los intentos.
- `include_raw=True` hace que LangChain no lance excepciones por una salida malformada, sino que las deje en el resultado. Por eso `check_output` las vuelve a lanzar, para que `with_retry` las vea y para poder leer el `finish_reason`.
- El esquema valida la estructura (campos obligatorios, el enum de criticidad) y reglas de negocio (la lista de tecnologías no puede estar vacía y se limpian los repetidos).
- Los logs (`logging`) muestran cuándo se valida una salida, cuándo falla y por qué, y cuándo se agotan los reintentos.
- Con un texto ambiguo que no menciona ninguna tecnología, el modelo devuelve `"no identificada"` en lugar de inventar una.
