"""Corre el orquestador con una consulta que necesita investigación, análisis y síntesis.

python main.py
python main.py "Compará las reseñas del Teclado Forja K1 y de los Auriculares Aurora X"
python main.py --provider anthropic
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from agents.supervisor import Limits
from config import BASE_DIR, Settings
from graph import Models, build_graph, build_model
from runner import run_question

TRACE_PATH = BASE_DIR / "trazas" / "ejemplo_traza.json"

DEFAULT_QUESTION = (
    "Quiero comprar auriculares o un parlante. Investigá las reseñas de los Auriculares Aurora X "
    "y del Parlante Brisa Mini, analizá el sentimiento y el puntaje promedio de cada uno y "
    "recomendame cuál conviene."
)


def print_flow(result: dict[str, Any]) -> None:
    print(f"\nPregunta: {result['pregunta']}\n")
    for event in result["flujo"]:
        if event["nodo"] == "supervisor":
            print(f"[supervisor #{event['paso']}] -> {event['destino']}")
            print(f"    razón: {event['razon']}")
            if event["instruccion"]:
                print(f"    instrucción: {event['instruccion']}")
            if event["ajuste"]:
                print(f"    AJUSTE DE SEGURIDAD: {event['ajuste']}")
        else:
            tools = ", ".join(event["herramientas"]) or "sin herramientas"
            print(f"[{event['nodo']}] intento {event['intento']} ({tools})")
            print(f"    {event['resultado'][:300].replace(chr(10), ' ')}")
    print(f"\n=== Respuesta final ===\n{result['respuesta_final']}")
    if result["error"]:
        print(f"\nERROR: {result['error']}")


async def run(settings: Settings, question: str, trace_path: Path | None) -> int:
    model = build_model(settings)
    limits = Limits(max_steps=settings.max_steps, max_attempts=settings.max_attempts)
    graph = build_graph(Models.same(model), limits)
    result = await run_question(graph, question, recursion_limit=settings.recursion_limit)
    print_flow(result)

    if trace_path is not None:
        document = {
            "modelo": f"{settings.provider}:{settings.model}",
            "limites": {"max_steps": settings.max_steps, "max_attempts": settings.max_attempts},
            **result,
        }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nTraza guardada en {trace_path}")
    return 0 if result["completada"] and not result["error"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente")
    parser.add_argument("pregunta", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--traza", type=Path, default=TRACE_PATH, help="dónde guardar la traza")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env(args.provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        sys.exit(1)

    custom = args.pregunta != DEFAULT_QUESTION
    sys.exit(asyncio.run(run(settings, args.pregunta, None if custom else args.traza)))


if __name__ == "__main__":
    main()
