"""Demo del flujo human-in-the-loop contra la API.

python scripts/demo_hitl.py             # pide una acción crítica, espera y APRUEBA
python scripts/demo_hitl.py --rechazar  # igual, pero la rechaza

1. Pide un informe y que se envíe por email (acción con efectos secundarios).
2. El job queda en WAITING_APPROVAL: el email todavía no salió.
3. Una persona aprueba o rechaza con POST /tasks/{id}/approve.
"""

import argparse
import asyncio
import json
import sys

import httpx

QUESTION = (
    "Compará las reseñas de los Auriculares Aurora X y del Parlante Brisa Mini, recomendame "
    "uno y enviá el informe por email a gerencia@ejemplo.com."
)


async def wait_for(client: httpx.AsyncClient, job_id: str, *statuses: str) -> dict:
    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in statuses:
            return job
        await asyncio.sleep(1)


async def main(url: str, approve: bool) -> int:
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        job_id = (await client.post("/tasks", json={"pregunta": QUESTION})).json()["job_id"]
        print(f"Tarea creada: {job_id}")

        job = await wait_for(client, job_id, "WAITING_APPROVAL", "DONE", "FAILED")
        if job["status"] != "WAITING_APPROVAL":
            print(f"La tarea terminó sin pedir aprobación: {job['status']} {job.get('error')}")
            return 1
        print("\nEstado: WAITING_APPROVAL. Acción que espera aprobación:")
        accion = job["accion_pendiente"]
        print(f"  {accion['nombre']} -> {accion['destinatario']} | asunto: {accion['asunto']}")
        print(f"  (el cuerpo es el informe, {len(accion['cuerpo'])} caracteres)")
        print(f"Emails enviados hasta ahora: {len((await client.get('/outbox')).json())}")

        decision = {"aprobado": approve, "comentario": "revisado en la demo"}
        print(f"\n{'Aprobando' if approve else 'Rechazando'} la acción...")
        await client.post(f"/tasks/{job_id}/approve", json=decision)

        job = await wait_for(client, job_id, "DONE", "FAILED")
        print(f"\nEstado final: {job['status']}")
        print(f"Acciones ejecutadas: {json.dumps(job['acciones_ejecutadas'], ensure_ascii=False)}")
        print(f"Emails enviados: {len((await client.get('/outbox')).json())}")
    return 0 if job["status"] == "DONE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo del flujo human-in-the-loop")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rechazar", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main(args.url, not args.rechazar)))
