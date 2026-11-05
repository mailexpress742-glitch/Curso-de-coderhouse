"""Herramientas del agente. El modelo elige cuál usar leyendo SOLO el nombre y el docstring,
así que cada docstring dice qué hace, cuándo usarla y qué devuelve."""

import asyncio
from typing import Any

from langchain_core.tools import tool

from database import CLIENTES, buscar_pedido, normalizar, pedidos_de, total_pedido

# Simula la latencia de una consulta real a una base de datos.
LATENCY_S = 0.05


@tool
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """Busca clientes por nombre o apellido y devuelve su cliente_id.

    Usala cuando el usuario nombra a un cliente por su nombre ("Ana Pérez", "Lucía") y todavía
    no conocés su cliente_id. La búsqueda ignora mayúsculas y tildes y acepta nombres parciales.
    Devuelve {"clientes": [{"cliente_id": 104, "nombre": "Ana Pérez"}]}. Si hay más de una
    coincidencia, preguntale al usuario cuál es. Si no hay ninguna devuelve {"error": ...}.
    """
    await asyncio.sleep(LATENCY_S)
    buscado = normalizar(nombre)
    encontrados = [
        {"cliente_id": cliente_id, "nombre": completo}
        for cliente_id, completo in CLIENTES.items()
        if buscado and buscado in normalizar(completo)
    ]
    if not encontrados:
        return {"error": f"no hay clientes que coincidan con '{nombre}'"}
    return {"clientes": encontrados}


@tool
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Lista los pedidos de un cliente a partir de su cliente_id numérico.

    Usala para saber cuántos pedidos hizo un cliente y cuánto gastó en total. Devuelve
    {"cliente_id": 102, "cantidad_pedidos": 3, "total": 14500, "pedidos": [...]}, donde cada
    pedido tiene pedido_id, fecha y total, ordenados del más antiguo al más reciente (el último
    de la lista es el más reciente). No incluye los productos de cada pedido: para eso usá
    detalle_pedido con el pedido_id. Si el cliente no existe devuelve {"error": ...}; en ese
    caso, si el usuario dio un nombre, probá con buscar_cliente.
    """
    await asyncio.sleep(LATENCY_S)
    if cliente_id not in CLIENTES:
        return {"error": f"no existe el cliente {cliente_id}"}
    pedidos = pedidos_de(cliente_id)
    return {
        "cliente_id": cliente_id,
        "cantidad_pedidos": len(pedidos),
        "total": sum(total_pedido(p) for p in pedidos),
        "pedidos": [
            {"pedido_id": p["id"], "fecha": p["fecha"], "total": total_pedido(p)} for p in pedidos
        ],
    }


@tool
async def detalle_pedido(pedido_id: str) -> dict[str, Any]:
    """Devuelve el detalle de un pedido: estado, fecha, productos, cantidades y total.

    Usala cuando necesites saber qué compró el cliente en un pedido puntual o en qué estado
    está. Necesita el pedido_id con formato "P-2003", que se obtiene de buscar_pedidos. Si el
    pedido no existe devuelve {"error": ...}.
    """
    await asyncio.sleep(LATENCY_S)
    pedido = buscar_pedido(pedido_id)
    if pedido is None:
        return {"error": f"no existe el pedido '{pedido_id}'"}
    return {
        "pedido_id": pedido["id"],
        "cliente_id": pedido["cliente_id"],
        "fecha": pedido["fecha"],
        "estado": pedido["estado"],
        "items": pedido["items"],
        "total": total_pedido(pedido),
    }


TOOLS = [buscar_cliente, buscar_pedidos, detalle_pedido]
