"""Evalúa el recuperador con un golden set: Precision@k y Recall@k.

python evaluate.py
python evaluate.py --golden golden_set.json --k 5
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import BaseModel, ValidationError

from config import BASE_DIR, Settings
from hybrid import RAGSystem

Retrieve = Callable[[str], Awaitable[list[Document]]]


class GoldenItem(BaseModel):
    pregunta: str
    documento_id_esperado: str


@dataclass
class QuestionResult:
    question: str
    expected: str
    retrieved: list[str]
    precision: float
    precision_norm: float
    recall: float
    top1: float


def precision_at_k(
    retrieved: Sequence[str], expected: str, k: int, available: int | None = None
) -> float:
    """Fracción de los k recuperados que pertenecen al documento esperado.

    Si el documento esperado tiene menos de k fragmentos (`available`), nunca se puede
    llegar a 1.0. En ese caso se divide por `available`: es la precisión normalizada.
    """
    relevant = sum(source == expected for source in retrieved[:k])
    return relevant / min(k, available or k)


def recall_at_k(retrieved: Sequence[str], expected: str, k: int) -> float:
    """1 si el documento esperado aparece entre los k recuperados, 0 si no."""
    return 1.0 if expected in retrieved[:k] else 0.0


def load_golden_set(path: Path) -> list[GoldenItem]:
    return [GoldenItem.model_validate(item) for item in json.loads(path.read_text("utf-8"))]


async def evaluate(
    retrieve: Retrieve, golden: list[GoldenItem], k: int, chunk_counts: Mapping[str, int]
) -> list[QuestionResult]:
    results = []
    for item in golden:
        docs = await retrieve(item.pregunta)
        sources = [doc.metadata["source"] for doc in docs]
        results.append(
            QuestionResult(
                question=item.pregunta,
                expected=item.documento_id_esperado,
                retrieved=sources,
                precision=precision_at_k(sources, item.documento_id_esperado, k),
                precision_norm=precision_at_k(
                    sources, item.documento_id_esperado, k, chunk_counts[item.documento_id_esperado]
                ),
                recall=recall_at_k(sources, item.documento_id_esperado, k),
                top1=recall_at_k(sources, item.documento_id_esperado, 1),
            )
        )
    return results


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def print_report(
    results: dict[str, list[QuestionResult]], system: RAGSystem, golden: list[GoldenItem], k: int
) -> None:
    chunk_total = sum(system.chunks_per_source().values())

    print(f"\n=== Evaluación: {len(golden)} preguntas sobre {chunk_total} fragmentos (k={k}) ===")
    print(
        f"{'recuperador':<12} {f'Recall@{k}':>9} {f'Precision@{k}':>13} "
        f"{'Prec.norm.':>11} {'Recall@1':>9}"
    )
    for name, rows in results.items():
        print(
            f"{name:<12} {mean([r.recall for r in rows]):>9.2f} "
            f"{mean([r.precision for r in rows]):>13.2f} "
            f"{mean([r.precision_norm for r in rows]):>11.2f} {mean([r.top1 for r in rows]):>9.2f}"
        )
    print(
        "\nPrec.norm.: la precisión dividida por el máximo alcanzable (un documento con 2 "
        f"fragmentos no puede aportar más de 2 de los {k} resultados)."
    )

    print("\nDetalle del recuperador híbrido:")
    for row in results["híbrido"]:
        mark = "OK " if row.recall else "FALLÓ"
        print(f"  [{mark}] {row.question}")
        print(f"         esperado: {row.expected} | recuperados: {row.retrieved}")


async def run(golden_path: Path, k: int) -> int:
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print(f"Configuración inválida: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
        return 1

    settings = settings.model_copy(update={"top_k": k})
    golden = load_golden_set(golden_path)
    system = RAGSystem.from_settings(settings)

    counts = system.chunks_per_source()
    results = {
        "híbrido": await evaluate(system.retrieve, golden, k, counts),
        "vectorial": await evaluate(system.retrieve_vector, golden, k, counts),
        "BM25": await evaluate(system.retrieve_bm25, golden, k, counts),
    }
    print_report(results, system, golden, k)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del recuperador híbrido")
    parser.add_argument("--golden", type=Path, default=BASE_DIR / "golden_set.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    sys.exit(asyncio.run(run(args.golden, args.k)))


if __name__ == "__main__":
    main()
