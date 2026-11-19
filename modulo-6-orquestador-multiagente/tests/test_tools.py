import pytest

from agents.analyst_agent import (
    analizar_sentimiento,
    calcular_estadisticas,
    label_for,
    score_text,
    validar_resenas,
)
from agents.research_agent import buscar_ficha_producto, buscar_resenas, listar_productos
from knowledge import find_product


async def test_listar_productos_returns_all_products():
    result = await listar_productos.ainvoke({})
    assert len(result["productos"]) == 3


async def test_buscar_resenas_accepts_partial_names_and_any_case():
    result = await buscar_resenas.ainvoke({"producto": "aurora"})
    assert result["producto"] == "Auriculares Aurora X"
    assert result["cantidad"] == 6
    assert {"id", "puntaje", "texto"} <= set(result["resenas"][0])


async def test_unknown_product_is_an_error_with_a_hint():
    result = await buscar_resenas.ainvoke({"producto": "Mouse Zafiro"})
    assert "listar_productos" in result["error"]


async def test_buscar_ficha_producto_returns_price_and_specs():
    result = await buscar_ficha_producto.ainvoke({"producto": "teclado forja"})
    assert result["precio_ars"] == 61900
    assert result["ficha"]["switches"] == "lineales"


def test_find_product_returns_none_for_empty_name():
    assert find_product("  ") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Excelente sonido, muy recomiendo", "positivo"),
        ("Horrible, llegó defectuoso y pésima atención", "negativo"),
        ("Es un parlante", "neutro"),
    ],
)
def test_sentiment_labels(text, expected):
    assert label_for(score_text(text)) == expected


def test_a_negation_flips_the_word_it_precedes():
    assert score_text("es perfecto") > 0
    assert score_text("no es perfecto") < 0
    assert score_text("sin problemas") > 0


def test_sentiment_ignores_accents_and_case():
    assert score_text("EXCELENTE") == score_text("excelente")
    assert score_text("pésima") == score_text("pesima")


def test_sentiment_is_deterministic():
    assert score_text("buena batería pero frágil") == score_text("buena batería pero frágil")


def test_analizar_sentimiento_aggregates_percentages():
    result = analizar_sentimiento.invoke(
        {"textos": ["Excelente producto", "Horrible, muy malo", "Es un parlante", "Perfecto"]}
    )
    assert result["porcentaje_positivo"] == 50
    assert result["porcentaje_negativo"] == 25
    assert result["porcentaje_neutro"] == 25
    assert len(result["detalle"]) == 4


def test_analizar_sentimiento_empty_list_is_an_error():
    assert "error" in analizar_sentimiento.invoke({"textos": []})


def test_calcular_estadisticas():
    result = calcular_estadisticas.invoke({"valores": [5, 4, 3, 4]})
    assert result["promedio"] == 4.0
    assert result["mediana"] == 4
    assert (result["minimo"], result["maximo"]) == (3, 5)


def test_calcular_estadisticas_empty_list_is_an_error():
    assert "error" in calcular_estadisticas.invoke({"valores": []})


def test_validar_resenas_accepts_well_formed_reviews():
    result = validar_resenas.invoke(
        {"resenas": [{"id": "A-1", "puntaje": 5, "texto": "Muy bueno"}]}
    )
    assert result == {"validas": 1, "invalidas": 0, "errores": []}


def test_validar_resenas_reports_each_problem():
    result = validar_resenas.invoke(
        {
            "resenas": [
                {"id": "A-1", "puntaje": 9, "texto": "Muy bueno"},
                {"id": "A-2", "puntaje": 3},
                {"id": "A-3", "puntaje": 4, "texto": "Ok bien"},
            ]
        }
    )
    assert result["validas"] == 1
    assert result["invalidas"] == 2
    assert "puntaje" in result["errores"][0]
