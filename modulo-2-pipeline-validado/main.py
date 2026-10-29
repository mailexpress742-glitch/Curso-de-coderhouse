"""Prueba del pipeline de extracción técnica con dos textos.

python main.py                       # usa LLM_PROVIDER del .env
python main.py --provider anthropic
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from chain import build_chain, build_model, process_text
from config import Settings
from schemas import PipelineError, Provider

ARCHITECTURE_TEXT = (
    "Nuestra API en FastAPI guarda las sesiones en Redis y los pedidos en PostgreSQL. "
    "Desde ayer, cuando suben las conexiones concurrentes, PostgreSQL llega al límite de "
    "conexiones y la API devuelve 500 en el checkout."
)

AMBIGUOUS_TEXT = "Che, el sistema anda medio raro desde la actualización, fijate cuando puedas."


async def run_example(chain, title: str, text: str) -> bool:
    print(f"\n--- {title} ---")
    print(f"Texto: {text}")
    try:
        result = await process_text(text, chain)
    except PipelineError as exc:
        print(f"El pipeline no pudo validar la salida: {exc}")
        return False
    print(result.model_dump_json(indent=2))
    return True


async def run(provider: str | None) -> int:
    try:
        settings = Settings.from_env(provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        return 1

    print(f"Proveedor: {settings.provider} | modelo: {settings.model}")
    chain = build_chain(build_model(settings), max_attempts=settings.max_attempts)

    results = [
        await run_example(chain, "Descripción de arquitectura", ARCHITECTURE_TEXT),
        await run_example(chain, "Texto ambiguo (prueba de estrés)", AMBIGUOUS_TEXT),
    ]
    return 0 if all(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del pipeline de extracción técnica")
    parser.add_argument("--provider", choices=list(Provider), default=None)
    args = parser.parse_args()

    # Sin esto, la consola de Windows falla con caracteres que no están en cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Los logs de httpx imprimen cada request y tapan los del pipeline.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    sys.exit(asyncio.run(run(args.provider)))


if __name__ == "__main__":
    main()
