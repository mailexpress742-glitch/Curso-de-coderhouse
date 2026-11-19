"""Agente investigador: consulta la base de productos y reseñas. No analiza ni calcula."""

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from knowledge import find_product, load_products

LATENCY_S = 0.05

SYSTEM_PROMPT = """Sos el investigador de un equipo. Tu único trabajo es buscar información en \
la base de productos con tus herramientas y devolverla.

- Consultá la base para cada producto que te pidan. No inventes datos.
- Si un producto no existe, probá con listar_productos para ver los nombres válidos.
- Respondé con un resumen breve de lo que encontraste (producto, precio, cantidad de reseñas). \
Los datos completos de las herramientas se le pasan al analista aparte: no hace falta que \
copies las reseñas.
- No analices ni saques conclusiones: eso lo hace el analista."""


@tool
async def listar_productos() -> dict[str, Any]:
    """Lista los productos que existen en la base, con su categoría y precio en pesos.

    Usala cuando no estés seguro del nombre exacto de un producto o cuando una búsqueda
    devuelva un error. Devuelve {"productos": [{"producto": ..., "categoria": ...,
    "precio_ars": ...}]}.
    """
    await asyncio.sleep(LATENCY_S)
    return {
        "productos": [
            {"producto": p["producto"], "categoria": p["categoria"], "precio_ars": p["precio_ars"]}
            for p in load_products()
        ]
    }


@tool
async def buscar_ficha_producto(producto: str) -> dict[str, Any]:
    """Devuelve la ficha técnica y el precio de un producto a partir de su nombre.

    Usala para conocer las características de un producto (batería, conectividad, peso) y su
    precio en pesos argentinos. Acepta nombres parciales como "aurora". Devuelve
    {"producto", "categoria", "precio_ars", "ficha"} o {"error": ...} si no existe.
    """
    await asyncio.sleep(LATENCY_S)
    found = find_product(producto)
    if found is None:
        return {"error": f"no existe el producto '{producto}'. Probá con listar_productos."}
    return {key: found[key] for key in ("producto", "categoria", "precio_ars", "ficha")}


@tool
async def buscar_resenas(producto: str) -> dict[str, Any]:
    """Devuelve todas las reseñas de clientes de un producto, cada una con su puntaje de 1 a 5.

    Usala cuando haya que conocer la opinión de los clientes sobre un producto. Acepta nombres
    parciales como "brisa". Devuelve {"producto", "cantidad", "resenas": [{"id", "puntaje",
    "texto"}]} o {"error": ...} si el producto no existe.
    """
    await asyncio.sleep(LATENCY_S)
    found = find_product(producto)
    if found is None:
        return {"error": f"no existe el producto '{producto}'. Probá con listar_productos."}
    return {
        "producto": found["producto"],
        "cantidad": len(found["resenas"]),
        "resenas": found["resenas"],
    }


RESEARCH_TOOLS = [listar_productos, buscar_ficha_producto, buscar_resenas]


def build_research_agent(model: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(model, RESEARCH_TOOLS, prompt=SYSTEM_PROMPT, name="investigador")
