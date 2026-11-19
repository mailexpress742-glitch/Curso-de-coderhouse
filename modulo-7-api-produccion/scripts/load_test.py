"""Prueba de carga: lanza 5 peticiones concurrentes contra la API y mide cada una.

python scripts/load_test.py
python scripts/load_test.py --url http://localhost:8000 --concurrency 5

Imprime la latencia de cada tarea (desde que se envía hasta que termina) y sus percentiles.
Los tokens y el costo por ejecución los calcula Phoenix a partir de las trazas.
"""

import argparse
import asyncio
import math
import statistics
import sys
import time

import httpx

QUESTIONS = [
    "Compará las reseñas de los Auriculares Aurora X y del Parlante Brisa Mini y recomendame uno.",
    "Analizá el sentimiento y el puntaje promedio de las reseñas del Teclado Forja K1.",
    "¿Qué producto tiene mejores reseñas: el Teclado Forja K1 o los Auriculares Aurora X?",
    "Investigá el Parlante Brisa Mini y decime si vale la pena según sus reseñas.",
    "Compará los tres productos de la base por puntaje promedio y sentimiento.",
]
TERMINAL = {"DONE", "FAILED"}


def percentile(values: list[float], q: float) -> float:
    """Percentil por el método del rango más cercano (nearest-rank)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


async def run_one(client: httpx.AsyncClient, index: int, question: str, poll_s: float) -> dict:
    started = time.perf_counter()
    response = await client.post("/tasks", json={"pregunta": question})
    response.raise_for_status()
    accepted = time.perf_counter() - started
    job_id = response.json()["job_id"]

    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in TERMINAL:
            break
        await asyncio.sleep(poll_s)
    return {
        "n": index,
        "job_id": job_id,
        "status": job["status"],
        "aceptada_s": accepted,
        "total_s": time.perf_counter() - started,
        "error": job.get("error"),
    }


async def main(url: str, concurrency: int, poll_s: float) -> int:
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(concurrency)]
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        await client.get("/health")
        started = time.perf_counter()
        results = await asyncio.gather(
            *[run_one(client, i + 1, q, poll_s) for i, q in enumerate(questions)]
        )
        wall = time.perf_counter() - started

    print(f"\n{concurrency} peticiones concurrentes contra {url}\n")
    print(f"{'#':>2} {'estado':<7} {'aceptada (s)':>13} {'total (s)':>10}  job_id")
    for row in results:
        print(
            f"{row['n']:>2} {row['status']:<7} {row['aceptada_s']:>13.3f} {row['total_s']:>10.1f}"
            f"  {row['job_id']}"
        )
        if row["error"]:
            print(f"   error: {row['error']}")

    totals = [row["total_s"] for row in results]
    print(f"\nTiempo total de la corrida: {wall:.1f} s")
    print(
        f"Latencia por tarea: min {min(totals):.1f} s | mediana {statistics.median(totals):.1f} s"
        f" | p95 {percentile(totals, 95):.1f} s | max {max(totals):.1f} s"
    )
    print(
        "El endpoint POST respondió en "
        f"{max(row['aceptada_s'] for row in results) * 1000:.0f} ms como máximo (solo encola)."
    )
    failed = [row for row in results if row["status"] != "DONE"]
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de carga de la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--poll", type=float, default=1.0, help="segundos entre consultas")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main(args.url, args.concurrency, args.poll)))
