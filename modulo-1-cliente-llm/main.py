"""Hace la misma pregunta en modo normal y en streaming.

python main.py                       # usa LLM_PROVIDER del .env
python main.py --provider anthropic
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from manager import AsyncLLMManager
from schemas import ChatMessage, GenerationParams, ModelResponse, Provider, Role

QUESTION = "¿Qué es la entropía?"

MESSAGES = [
    ChatMessage(role=Role.SYSTEM, content="Respondé en español, en un solo párrafo breve."),
    ChatMessage(role=Role.USER, content=QUESTION),
]


def format_metadata(response: ModelResponse) -> str:
    usage = response.usage
    tokens = f"{usage.input_tokens} in / {usage.output_tokens} out" if usage else "s/d"
    return (
        f"[modelo: {response.model} | fin: {response.finish_reason} | tokens: {tokens}"
        f" | latencia: {response.latency_ms:.0f} ms | intentos: {response.attempts}]"
    )


async def ask(llm: AsyncLLMManager) -> bool:
    print("\n--- Modo normal ---")
    response = await llm.generate(MESSAGES)
    if not response.ok:
        print(f"Error ({response.error.kind}): {response.error.message}")
        return False

    print(response.content)
    print(format_metadata(response))
    return True


async def ask_streaming(llm: AsyncLLMManager) -> bool:
    print("\n--- Modo streaming ---")
    async for chunk in llm.stream(MESSAGES):
        if chunk.error:
            print(f"\nError ({chunk.error.kind}): {chunk.error.message}")
            return False
        if chunk.done:
            tokens = chunk.usage.output_tokens if chunk.usage else "s/d"
            print(f"\n[fin: {chunk.finish_reason} | tokens de salida: {tokens}]")
        else:
            print(chunk.delta, end="", flush=True)
    return True


def show_validation() -> None:
    print("\n--- Validación ---")
    try:
        GenerationParams(temperature=3.5)
    except ValidationError as exc:
        print(f"temperature=3.5 rechazada sin llamar a la API: {exc.errors()[0]['msg']}")


async def run(provider: str | None) -> int:
    try:
        llm = AsyncLLMManager.from_env(provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá el archivo .env (hay un ejemplo en .env.example).")
        return 1

    async with llm:
        print(f"Proveedor: {llm.provider} | modelo: {llm.model}")
        print(f"Pregunta: {QUESTION}")
        normal_ok = await ask(llm)
        stream_ok = await ask_streaming(llm)
        show_validation()

    return 0 if normal_ok and stream_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del cliente unificado de LLMs")
    parser.add_argument("--provider", choices=list(Provider), default=None)
    args = parser.parse_args()

    # Sin esto, la consola de Windows falla con símbolos como Σ o ≈.
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    sys.exit(asyncio.run(run(args.provider)))


if __name__ == "__main__":
    main()
