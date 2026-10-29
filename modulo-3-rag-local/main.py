"""Indexa los documentos (si hace falta) y prueba el sistema con dos preguntas.

python main.py                         # las dos preguntas de prueba
python main.py "¿Qué es el overlap?"   # una pregunta propia
python main.py --provider anthropic
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from config import Settings
from ingest import run_ingest
from rag import RAGPipeline, get_rag_response
from schemas import EmbeddingMismatchError, Provider, RAGError

# Está en los documentos (data/02_chunking.md).
QUESTION_IN_DOCS = "¿Para qué sirve el overlap al fragmentar documentos?"
# Es sobre el mismo tema pero los documentos no lo mencionan: no debería inventar.
TRAP_QUESTION = "¿Cuánto cuesta por mes el plan de Pinecone y cuántos usuarios soporta?"


async def ask(pipeline: RAGPipeline, title: str, question: str) -> bool:
    print(f"\n--- {title} ---")
    print(f"Pregunta: {question}")
    try:
        response = await get_rag_response(question, pipeline)
    except RAGError as exc:
        print(f"Error: {exc}")
        return False

    print(f"Respuesta: {response.respuesta}")
    for source in response.fuentes:
        print(
            f"  fuente: {source.archivo} (fragmento {source.fragmento}, dist. {source.distancia})"
        )
    print(f"encontrada={response.encontrada}")
    return True


async def run(provider: str | None, questions: list[str]) -> int:
    try:
        settings = Settings.from_env(provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        return 1

    print(f"Proveedor: {settings.provider} | modelo: {settings.model}")
    print(f"Embeddings: {settings.embeddings_provider}")
    try:
        report = await asyncio.to_thread(run_ingest, settings)
        pipeline = RAGPipeline.from_settings(settings)
    except EmbeddingMismatchError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Base lista: {report.added} fragmentos nuevos, {report.unchanged} ya estaban indexados.")

    if questions:
        results = [await ask(pipeline, "Pregunta", q) for q in questions]
    else:
        results = [
            await ask(pipeline, "Pregunta con respuesta en los documentos", QUESTION_IN_DOCS),
            await ask(pipeline, "Pregunta trampa (no está en los documentos)", TRAP_QUESTION),
        ]
    return 0 if all(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del sistema RAG local")
    parser.add_argument("questions", nargs="*", help="preguntas propias (opcional)")
    parser.add_argument("--provider", choices=list(Provider), default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    sys.exit(asyncio.run(run(args.provider, args.questions)))


if __name__ == "__main__":
    main()
