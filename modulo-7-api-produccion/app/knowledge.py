"""Base de conocimiento de productos y reseñas (archivo JSON local, simula una fuente externa)."""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parent / "data" / "productos.json"


def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para comparar nombres y palabras."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


@lru_cache(maxsize=1)
def load_products() -> list[dict[str, Any]]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def find_product(name: str) -> dict[str, Any] | None:
    """Busca un producto por nombre: acepta nombres parciales ("aurora") y cualquier orden."""
    words = normalize(name).split()
    if not words:
        return None
    for product in load_products():
        title = normalize(product["producto"])
        if all(word in title for word in words):
            return product
    return None
