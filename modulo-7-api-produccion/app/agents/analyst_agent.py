"""Agente analista: procesa los datos que obtuvo el investigador (valida, mide sentimiento y
calcula estadísticas). No busca información por su cuenta."""

import statistics
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError

from app.knowledge import normalize

SYSTEM_PROMPT = """Sos el analista de un equipo. Procesás los datos que ya reunió el investigador \
con tus herramientas de cálculo. No buscás información nueva.

- Los datos crudos del investigador están en "Aportes de otros agentes", en JSON. Usalos tal \
cual como argumentos de tus herramientas; no los reescribas ni los resumas.
- Antes de analizar reseñas, validalas con validar_resenas. Si hay errores, informalos.
- Hacé todos los cálculos con las herramientas, nunca de memoria.
- Respondé con los resultados numéricos por producto (puntaje promedio, sentimiento, \
porcentajes) y una comparación breve basada en esos números."""

POSITIVE_WORDS = {
    "excelente", "excelentes", "bueno", "buena", "buenos", "buenas", "mejor", "mejores",
    "fantastica", "fantastico", "increible", "maravilla", "perfecto", "perfecta", "impecable",
    "hermosa", "hermoso", "lindo", "linda", "comodo", "comodos", "estable", "potente",
    "recomiendo", "vale", "pena", "facil", "simple", "solida", "solido", "silencioso", "rinde",
    "bonitos", "bonito", "dura", "resiste", "correcto", "cumple",
}  # fmt: skip
NEGATIVE_WORDS = {
    "malo", "mala", "malos", "malas", "horrible", "pesima", "pesimo", "decepciono", "defectuoso",
    "distorsiona", "traba", "frágil", "fragil", "rayo", "caro", "pesado", "ensucian", "tarda",
    "desconectan", "tardo", "peor", "problema", "problemas",
}  # fmt: skip
POSITIVE = {normalize(word) for word in POSITIVE_WORDS}
NEGATIVE = {normalize(word) for word in NEGATIVE_WORDS}
NEGATIONS = {"no", "nunca", "jamas", "sin", "ni"}


class Resena(BaseModel):
    """Contrato de una reseña: se usa para validar los datos antes de analizarlos."""

    id: str
    puntaje: int = Field(ge=1, le=5)
    texto: str = Field(min_length=3)


def score_text(text: str) -> float:
    """Sentimiento de un texto entre -1 (muy negativo) y 1 (muy positivo), con un léxico simple.

    Una negación ("no", "nunca", "sin") invierte el signo de una de las dos palabras que siguen.
    """
    words = normalize(text).replace(",", " ").replace(".", " ").split()
    positive = negative = 0
    for index, word in enumerate(words):
        if word in POSITIVE:
            polarity = 1
        elif word in NEGATIVE:
            polarity = -1
        else:
            continue
        if any(previous in NEGATIONS for previous in words[max(0, index - 2) : index]):
            polarity = -polarity
        if polarity > 0:
            positive += 1
        else:
            negative += 1
    total = positive + negative
    return 0.0 if total == 0 else round((positive - negative) / total, 2)


def label_for(score: float) -> str:
    if score > 0.2:
        return "positivo"
    if score < -0.2:
        return "negativo"
    return "neutro"


@tool
def validar_resenas(resenas: list[dict[str, Any]]) -> dict[str, Any]:
    """Valida que una lista de reseñas tenga el formato correcto antes de analizarla.

    Usala primero, con las reseñas que trajo el investigador. Cada reseña debe tener "id",
    "puntaje" (entero de 1 a 5) y "texto". Devuelve {"validas": N, "invalidas": M,
    "errores": [...]} para que sepas si podés confiar en los datos.
    """
    errors = []
    valid = 0
    for index, item in enumerate(resenas):
        try:
            Resena.model_validate(item)
            valid += 1
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append(f"reseña {index}: {first['loc']} {first['msg']}")
    return {"validas": valid, "invalidas": len(errors), "errores": errors}


@tool
def analizar_sentimiento(textos: list[str]) -> dict[str, Any]:
    """Mide el sentimiento (positivo, negativo o neutro) de una lista de textos en español.

    Usala con los textos de las reseñas de UN producto a la vez. Devuelve el puntaje de
    sentimiento promedio entre -1 y 1, el porcentaje de textos positivos, negativos y
    neutros, y el detalle por texto. Es un cálculo con un léxico fijo: el mismo texto siempre
    da el mismo resultado.
    """
    if not textos:
        return {"error": "la lista de textos está vacía"}
    scores = [score_text(text) for text in textos]
    labels = [label_for(score) for score in scores]
    return {
        "sentimiento_promedio": round(statistics.mean(scores), 2),
        "porcentaje_positivo": round(100 * labels.count("positivo") / len(labels)),
        "porcentaje_negativo": round(100 * labels.count("negativo") / len(labels)),
        "porcentaje_neutro": round(100 * labels.count("neutro") / len(labels)),
        "detalle": [{"puntaje": s, "etiqueta": e} for s, e in zip(scores, labels, strict=True)],
    }


@tool
def calcular_estadisticas(valores: list[float]) -> dict[str, Any]:
    """Calcula estadísticas básicas de una lista de números.

    Usala, por ejemplo, con los puntajes de 1 a 5 de las reseñas de un producto. Devuelve
    cantidad, promedio, mediana, mínimo, máximo y desvío estándar. Si la lista está vacía
    devuelve {"error": ...}.
    """
    if not valores:
        return {"error": "la lista de valores está vacía"}
    return {
        "cantidad": len(valores),
        "promedio": round(statistics.mean(valores), 2),
        "mediana": statistics.median(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "desvio": round(statistics.pstdev(valores), 2),
    }


ANALYST_TOOLS = [validar_resenas, analizar_sentimiento, calcular_estadisticas]


def build_analyst_agent(model: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(model, ANALYST_TOOLS, prompt=SYSTEM_PROMPT, name="analista")
