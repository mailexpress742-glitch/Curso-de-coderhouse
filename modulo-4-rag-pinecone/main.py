"""Consulta el recuperador híbrido y muestra los top-5 fragmentos.

python main.py
python main.py "¿Cómo oculto una contraseña en la configuración?"
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from config import Settings
from hybrid import RAGSystem

DEFAULT_QUERY = "¿Qué herramienta reescribe automáticamente el código para migrar a Pydantic 2?"


async def run(query: str) -> int:
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        return 1

    system = RAGSystem.from_settings(settings)
    print(f"Consulta: {query}\n")
    for rank, doc in enumerate(await system.retrieve(query), start=1):
        meta = doc.metadata
        page = int(meta["page"])  # Pinecone devuelve los números como float
        print(f"{rank}. {meta['source']} | sección {page} | categoría: {meta['categoria']}")
        print(f"   etiquetas: {', '.join(meta['etiquetas'])}")
        print(f"   {' '.join(doc.page_content.split())[:160]}...")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta al recuperador híbrido")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    sys.exit(asyncio.run(run(args.query)))


if __name__ == "__main__":
    main()
