"""Corre el agente con una conversación de ejemplo y guarda la traza ReAct en un JSON.

python main.py                       # conversación de ejemplo (3 hilos)
python main.py --reset               # borra la memoria guardada antes de empezar
python main.py --thread demo-102 --pregunta "¿Y el primero que hizo?"   # un hilo existente
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import ValidationError

from agent import TurnResult, ask
from config import BASE_DIR, Settings
from graph import build_graph, build_model

TRACE_PATH = BASE_DIR / "trazas" / "ejemplo_traza.json"

# Cada hilo es una conversación independiente. El primero tiene seguimiento ("¿y el último?"),
# así que prueba que el agente recuerda el contexto con el mismo thread_id.
SCENARIOS: dict[str, list[str]] = {
    "demo-102": [
        "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?",
        "¿Y el último? ¿Qué productos incluía y en qué estado está?",
    ],
    "demo-nombre": ["¿Cuánto gastó en total Ana Pérez?"],
    "demo-error": ["¿Cuántos pedidos tuvo el cliente 999?"],
}


def print_result(thread_id: str, result: TurnResult) -> None:
    print(f"\n[{thread_id}] Usuario: {result.pregunta}")
    for step in result.pasos:
        if step["tipo"] == "accion":
            calls = ", ".join(f"{h['nombre']}({h['argumentos']})" for h in step["herramientas"])
            print(f"   -> acción: {calls}")
        elif step["tipo"] == "observacion":
            print(f"   <- {step['herramienta']}: {step['resultado']}")
    print(f"Agente: {result.respuesta}")
    print(f"   (herramientas usadas: {result.llamadas_a_herramientas})")
    if result.error:
        print(f"   ERROR: {result.error}")


async def run(settings: Settings, scenarios: dict[str, list[str]], trace_path: Path | None) -> int:
    model = build_model(settings)
    recorded: list[dict] = []

    # El checkpointer guarda el estado de cada thread_id en un archivo SQLite.
    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as checkpointer:
        graph = build_graph(model, checkpointer, max_context_messages=settings.max_context_messages)
        for thread_id, questions in scenarios.items():
            for question in questions:
                result = await ask(
                    graph, question, thread_id, recursion_limit=settings.recursion_limit
                )
                print_result(thread_id, result)
                recorded.append({"thread_id": thread_id, **result.to_dict()})

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "modelo": f"{settings.provider}:{settings.model}",
            "recursion_limit": settings.recursion_limit,
            "turnos": recorded,
        }
        trace_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nTraza guardada en {trace_path}")
    return 0 if not any(turn["error"] for turn in recorded) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente ReAct con memoria persistente")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--thread", default=None, help="thread_id de la conversación")
    parser.add_argument("--pregunta", default=None, help="una pregunta suelta (requiere --thread)")
    parser.add_argument("--reset", action="store_true", help="borra la memoria guardada")
    parser.add_argument("--traza", type=Path, default=TRACE_PATH, help="dónde guardar la traza")
    args = parser.parse_args()

    if bool(args.pregunta) != bool(args.thread):
        parser.error("--pregunta y --thread se usan juntos")

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env(args.provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        sys.exit(1)

    if args.reset and settings.db_path.exists():
        settings.db_path.unlink()
        print("Memoria borrada.")

    if args.pregunta:
        scenarios, trace_path = {args.thread: [args.pregunta]}, None
    else:
        scenarios, trace_path = SCENARIOS, args.traza
    sys.exit(asyncio.run(run(settings, scenarios, trace_path)))


if __name__ == "__main__":
    main()
