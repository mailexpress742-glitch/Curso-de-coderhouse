"""Saca las capturas del dashboard de Phoenix y las guarda en screenshots/.

python scripts/capture_screenshots.py
python scripts/capture_screenshots.py --project orquestador-multiagente --hitl-project hitl-demo

Usa Playwright con el Edge o el Chrome que ya tenga instalado la máquina (no descarga un
navegador). Hay que correr antes la prueba de carga, para que haya trazas que mostrar.
"""

import argparse
import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path

from playwright.async_api import Page, async_playwright

OUT = Path(__file__).resolve().parent.parent / "screenshots"
HIDDEN_COLUMNS = [
    "input",
    "output",
    "error",
    "user",
    "annotations",
    "trace annotations",
    "start time",
]
SHOWN_COLUMNS = ["latency", "total tokens", "total cost"]


def project_id(base: str, name: str) -> str:
    query = json.dumps({"query": "{ projects(first:50){edges{node{id name}}} }"}).encode()
    request = urllib.request.Request(f"{base}/graphql", query, {"content-type": "application/json"})
    with urllib.request.urlopen(request) as response:
        edges = json.load(response)["data"]["projects"]["edges"]
    for edge in edges:
        if edge["node"]["name"] == name:
            return edge["node"]["id"]
    raise SystemExit(f"no existe el proyecto '{name}' en Phoenix")


async def settle(page: Page, ms: int = 4000) -> None:
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(ms)


async def set_columns(page: Page) -> None:
    """Deja visibles solo las columnas que interesan: latencia, tokens y costo por traza."""
    await page.get_by_role("button", name="Columns").click()
    await page.wait_for_timeout(500)
    wanted = {label: False for label in HIDDEN_COLUMNS} | {label: True for label in SHOWN_COLUMNS}
    for label, visible in wanted.items():
        row = page.locator("label").filter(has_text=re.compile(rf"^\s*{re.escape(label)}\s*$"))
        checkbox = row.locator("input[type=checkbox]").first
        if await checkbox.is_checked() != visible:
            await row.first.click()
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(800)


async def hover_latency_chart(page: Page) -> None:
    """Pasa el mouse sobre el gráfico de percentiles hasta que aparezca el tooltip con P95."""
    chart = page.get_by_text("Trace latency").first.locator("xpath=ancestor::div[3]")
    box = await chart.bounding_box()
    if box is None:
        return
    before = await page.get_by_text("P95").count()
    for step in range(0, 60):
        x = box["x"] + box["width"] * (0.55 + step * 0.0075)
        for y_ratio in (0.35, 0.45, 0.55):
            await page.mouse.move(x, box["y"] + box["height"] * y_ratio)
            await page.wait_for_timeout(60)
            if await page.get_by_text("P95").count() > before:
                return


async def capture(base: str, project: str, hitl_project: str) -> None:
    OUT.mkdir(exist_ok=True)
    main_id = project_id(base, project)

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(channel="msedge", headless=True)
        except Exception:
            browser = await playwright.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page(viewport={"width": 1600, "height": 1000})

        await page.goto(f"{base}/projects?timeRangeKey=7d")
        await settle(page, 3000)
        await page.screenshot(path=OUT / "01_proyectos.png")

        await page.goto(f"{base}/projects/{main_id}/metrics?timeRangeKey=7d")
        await settle(page, 5000)
        await page.set_viewport_size({"width": 1600, "height": 1500})
        await page.wait_for_timeout(1500)
        await page.screenshot(path=OUT / "02_metricas_costo_latencia.png")
        await page.set_viewport_size({"width": 1600, "height": 1000})

        await page.goto(f"{base}/projects/{main_id}/traces?timeRangeKey=7d")
        await settle(page, 4000)
        await set_columns(page)
        await page.screenshot(path=OUT / "04_trazas_costo_tokens_latencia.png")

        await hover_latency_chart(page)
        await page.screenshot(path=OUT / "03_latencia_percentiles_p95.png")
        await page.mouse.move(5, 5)

        await page.locator("a", has_text="job.ejecutar").first.click()
        await settle(page, 5000)
        await page.screenshot(path=OUT / "05_traza_detalle.png")

        await page.goto(f"{base}/projects/{main_id}/sessions?timeRangeKey=7d")
        await settle(page, 3000)
        await page.screenshot(path=OUT / "06_sesiones_por_job.png")

        hitl_id = project_id(base, hitl_project)
        await page.goto(f"{base}/projects/{hitl_id}/traces?timeRangeKey=7d")
        await settle(page, 4000)
        await set_columns(page)
        await page.screenshot(path=OUT / "07_hitl_trazas.png")

        await browser.close()
    print(f"Capturas guardadas en {OUT}:")
    for path in sorted(OUT.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capturas del dashboard de Phoenix")
    parser.add_argument("--url", default="http://localhost:6006")
    parser.add_argument("--project", default="orquestador-multiagente")
    parser.add_argument("--hitl-project", default="hitl-demo")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(capture(args.url, args.project, args.hitl_project))
