"""Script de validación: hace la misma pregunta en modo normal y en streaming.

Uso:
    python main.py                       # usa LLM_PROVIDER del .env
    python main.py --provider anthropic  # fuerza un proveedor
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


def print_metadata(response: ModelResponse) -> None:
    usage = response.usage
    tokens = f"{usage.input_tokens} in / {usage.output_tokens} out" if usage else "s/d"
    print(
        f"[modelo: {response.model} | fin: {response.finish_reason} | tokens: {tokens}"
        f" | latencia: {response.latency_ms:.0f} ms | intentos: {response.attempts}]"
    )


async def demo_normal(llm: AsyncLLMManager) -> bool:
    print("\n=== 1) Modo normal ===")
    response = await llm.generate(MESSAGES)
    if not response.ok:
        print(f"Error controlado ({response.error.kind}): {response.error.message}")
        return False
    print(response.content)
    print_metadata(response)
    return True


async def demo_streaming(llm: AsyncLLMManager) -> bool:
    print("\n=== 2) Modo streaming ===")
    async for chunk in llm.stream(MESSAGES):
        if chunk.error:
            print(f"\nError controlado ({chunk.error.kind}): {chunk.error.message}")
            return False
        if chunk.done:
            tokens = chunk.usage.output_tokens if chunk.usage else "s/d"
            print(f"\n[fin: {chunk.finish_reason} | tokens de salida: {tokens}]")
        else:
            print(chunk.delta, end="", flush=True)
    return True


def demo_validation() -> None:
    print("\n=== 3) Validación con Pydantic ===")
    try:
        GenerationParams(temperature=3.5)
    except ValidationError as exc:
        print(f"temperature=3.5 rechazada antes de llamar a la API: {exc.errors()[0]['msg']}")


async def run(provider: str | None) -> int:
    try:
        llm = AsyncLLMManager.from_env(provider)
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['msg']}")
        print("Revisá tu archivo .env (ver .env.example).")
        return 1

    async with llm:
        print(f"Proveedor: {llm.provider} | modelo: {llm.model}")
        print(f"Pregunta: {QUESTION}")
        ok_normal = await demo_normal(llm)
        ok_stream = await demo_streaming(llm)
        demo_validation()
    return 0 if ok_normal and ok_stream else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del Unified Async LLM Client")
    parser.add_argument("--provider", choices=[p.value for p in Provider], default=None)
    args = parser.parse_args()

    # La salida del modelo puede traer símbolos (Σ, ≈, ...) que la consola de
    # Windows no codifica por defecto.
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    sys.exit(asyncio.run(run(args.provider)))


if __name__ == "__main__":
    main()
