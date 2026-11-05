# Agente de razonamiento cíclico con memoria persistente

Pre-entrega del módulo 5 de AI Engineering (Coderhouse).

Agente ReAct hecho con LangGraph que responde preguntas sobre clientes y pedidos consultando una base de datos simulada. El modelo decide solo cuándo usar una herramienta, mira el resultado y decide si necesita otra. La conversación se guarda en SQLite, así que con el mismo `thread_id` recuerda lo anterior, incluso después de reiniciar el programa.

```
Usuario: ¿Cuánto gastó en total Ana Pérez?
  -> acción:        buscar_cliente(nombre="Ana Pérez")
  <- observación:   {"clientes": [{"cliente_id": 104, "nombre": "Ana Pérez"}]}
  -> acción:        buscar_pedidos(cliente_id=104)
  <- observación:   {"cantidad_pedidos": 2, "total": 11600, ...}
Agente: Ana Pérez gastó $11.600 en total, sumando 2 pedidos.
```

## Instalación

Hace falta Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Después copiar `.env.example` como `.env` y completar la API key del proveedor que se vaya a usar (`LLM_PROVIDER` elige entre `openai` y `anthropic`).

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | | Obligatoria la del proveedor elegido |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `gpt-5-mini` / `claude-opus-5-5` | Modelo de cada proveedor |
| `CHECKPOINT_DB` | `checkpoints.sqlite` | Archivo donde se guarda la memoria |
| `AGENT_RECURSION_LIMIT` | `12` | Máximo de pasos del grafo por pregunta |
| `AGENT_MAX_CONTEXT_MESSAGES` | `30` | Mensajes que se le mandan al modelo como máximo |

## Uso

Conversación de ejemplo (3 hilos). Imprime cada paso y guarda la traza en `trazas/ejemplo_traza.json`:

```bash
python main.py --reset          # --reset borra la memoria guardada antes de empezar
python main.py --provider anthropic
```

Para seguir una conversación que ya existe, con el mismo `thread_id`. Funciona en un proceso nuevo porque la memoria está en el archivo SQLite:

```bash
python main.py --thread demo-102 --pregunta "¿Y el primero que hizo? ¿Qué compró?"
```

Los tests no usan API keys ni red: reemplazan el modelo por uno que sigue un guion.

```bash
pytest
```

## Estructura

```
database.py    clientes y pedidos simulados (en memoria)
tools.py       las 3 herramientas, con @tool y docstrings descriptivos
graph.py       AgentState, el StateGraph y el modelo
agent.py       ask(): ejecuta una pregunta y registra la traza ReAct
config.py      configuración desde variables de entorno
main.py        conversación de ejemplo
trazas/        ejemplo_traza.json: traza de una ejecución real
tests/
```

## Cómo funciona

**Herramientas (`tools.py`).** Son tres, definidas con `@tool` y asíncronas:

- `buscar_cliente(nombre)`: devuelve el `cliente_id` a partir de un nombre.
- `buscar_pedidos(cliente_id)`: cantidad de pedidos, total gastado y lista de pedidos.
- `detalle_pedido(pedido_id)`: productos, estado y total de un pedido.

El modelo elige cuál usar leyendo solo el nombre y el docstring, así que cada docstring explica qué hace, cuándo usarla, qué devuelve y qué hacer si da error. Los errores no lanzan excepciones: vuelven como `{"error": ...}` para que el modelo los vea y reaccione.

**Grafo (`graph.py`).** `AgentState` hereda de `MessagesState`, cuyo reducer `add_messages` agrega los mensajes nuevos al historial en lugar de pisarlo.

```
START -> agent --(tools_condition)--> tools -> agent -> ... -> END
```

- El nodo `agent` llama al modelo, que quedó vinculado a las herramientas con `bind_tools()`.
- La arista condicional `tools_condition` revisa el último mensaje: si el modelo pidió una herramienta va al nodo `tools` (un `ToolNode`), y si no, el grafo termina.
- El nodo `tools` ejecuta la herramienta y vuelve a `agent`. Ese retorno cierra el ciclo ReAct y permite que el agente use el resultado para decidir su próximo paso. No hay ningún `if/else` que elija herramientas: la decisión es del modelo.

**Ciclo de retorno.** Si una herramienta devuelve un error, el modelo lo recibe como una observación más. El prompt le pide que lo corrija con otra herramienta (por ejemplo buscar al cliente por nombre) o que le pida la aclaración al usuario. En la corrida de ejemplo, ante el cliente 999 (que no existe), el agente pidió el nombre del cliente en lugar de inventar un resultado.

**Memoria (`AsyncSqliteSaver`).** El grafo se compila con un checkpointer que guarda el estado de cada `thread_id` en un archivo SQLite. Como el programa es asíncrono se usa `AsyncSqliteSaver`, la versión `async` de `SqliteSaver`. Con el mismo `thread_id` el agente recuerda las preguntas anteriores ("¿y el último?"), y con otro `thread_id` empieza desde cero.

**Control de costos y de estado**

- `recursion_limit` (12 por defecto) se pasa en cada invocación. Si el agente entra en un bucle, el grafo se corta y `ask()` devuelve un error en lugar de seguir gastando.
- El estado guardado conserva toda la conversación, pero al modelo solo se le envían los últimos mensajes (`AGENT_MAX_CONTEXT_MESSAGES`). La ventana siempre empieza en un mensaje del usuario, para no separar el pedido de una herramienta de su resultado.

## Traza de ejecución

`trazas/ejemplo_traza.json` es la salida de una ejecución real de `python main.py`. Cada turno guarda la pregunta, la respuesta y los pasos del ciclo: `accion` (qué herramienta pidió el modelo y con qué argumentos), `observacion` (qué devolvió) y `respuesta`.

Resumen de la traza, con el modelo de Anthropic:

| Hilo | Pregunta | Herramientas | Qué muestra |
|---|---|---|---|
| `demo-102` | ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total? | `buscar_pedidos` | 3 pedidos, $14.500 |
| `demo-102` | ¿Y el último? ¿Qué productos incluía y en qué estado está? | `detalle_pedido` | Usa la memoria: sabe que "el último" es P-2003 |
| `demo-nombre` | ¿Cuánto gastó en total Ana Pérez? | `buscar_cliente`, `buscar_pedidos` | Razonamiento multi-paso: el resultado de la primera herramienta es el dato de la segunda |
| `demo-error` | ¿Cuántos pedidos tuvo el cliente 999? | `buscar_pedidos` | La herramienta devuelve un error y el agente pide una aclaración |

## Criterios de la consigna

| Criterio | Dónde |
|---|---|
| Autonomía | `graph.py`: el modelo decide con `bind_tools()` y `tools_condition`, sin rutas manuales |
| Ciclo de retorno | `tools -> agent` y los errores como observaciones; test `test_tool_error_goes_back_to_the_model_which_tries_again` |
| Resiliencia de estado | `AsyncSqliteSaver` + `thread_id`; test que reabre el archivo SQLite |
| Multi-paso (≥ 2 llamadas) | hilo `demo-nombre` de la traza |
| `recursion_limit` | `config.py` y `agent.py`; test `test_recursion_limit_stops_a_loop_without_raising` |
| Código limpio | Python 3.12, type hints, `asyncio`, ruff |
