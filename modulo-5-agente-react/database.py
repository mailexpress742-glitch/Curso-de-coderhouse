"""Base de datos simulada de clientes y pedidos. Vive en memoria: no hay que instalar nada."""

import unicodedata
from typing import TypedDict


class Item(TypedDict):
    producto: str
    cantidad: int
    precio_unitario: int


class Pedido(TypedDict):
    id: str
    cliente_id: int
    fecha: str
    estado: str
    items: list[Item]


CLIENTES: dict[int, str] = {
    101: "Martín Gómez",
    102: "Lucía Fernández",
    103: "Carlos Ruiz",
    104: "Ana Pérez",
}

PEDIDOS: list[Pedido] = [
    {
        "id": "P-1001",
        "cliente_id": 101,
        "fecha": "2026-09-02",
        "estado": "entregado",
        "items": [{"producto": "Cargador USB-C", "cantidad": 2, "precio_unitario": 4500}],
    },
    {
        "id": "P-2001",
        "cliente_id": 102,
        "fecha": "2026-08-03",
        "estado": "entregado",
        "items": [{"producto": "Teclado mecánico", "cantidad": 1, "precio_unitario": 3200}],
    },
    {
        "id": "P-2002",
        "cliente_id": 102,
        "fecha": "2026-09-14",
        "estado": "entregado",
        "items": [{"producto": "Monitor 24 pulgadas", "cantidad": 1, "precio_unitario": 5800}],
    },
    {
        "id": "P-2003",
        "cliente_id": 102,
        "fecha": "2026-10-27",
        "estado": "en camino",
        "items": [
            {"producto": "Auriculares inalámbricos", "cantidad": 1, "precio_unitario": 3500},
            {"producto": "Mouse ergonómico", "cantidad": 1, "precio_unitario": 2000},
        ],
    },
    {
        "id": "P-4001",
        "cliente_id": 104,
        "fecha": "2026-07-11",
        "estado": "entregado",
        "items": [{"producto": "Webcam HD", "cantidad": 1, "precio_unitario": 6100}],
    },
    {
        "id": "P-4002",
        "cliente_id": 104,
        "fecha": "2026-10-05",
        "estado": "entregado",
        "items": [{"producto": "Soporte para notebook", "cantidad": 2, "precio_unitario": 2750}],
    },
]


def total_pedido(pedido: Pedido) -> int:
    return sum(item["cantidad"] * item["precio_unitario"] for item in pedido["items"])


def pedidos_de(cliente_id: int) -> list[Pedido]:
    """Pedidos de un cliente, del más antiguo al más reciente."""
    return sorted((p for p in PEDIDOS if p["cliente_id"] == cliente_id), key=lambda p: p["fecha"])


def buscar_pedido(pedido_id: str) -> Pedido | None:
    return next((p for p in PEDIDOS if p["id"].lower() == pedido_id.strip().lower()), None)


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar nombres."""
    decomposed = unicodedata.normalize("NFKD", texto.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
