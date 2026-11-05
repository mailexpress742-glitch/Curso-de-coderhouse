import pytest

from tools import TOOLS, buscar_cliente, buscar_pedidos, detalle_pedido


async def test_buscar_cliente_ignores_case_and_accents():
    result = await buscar_cliente.ainvoke({"nombre": "ana perez"})
    assert result == {"clientes": [{"cliente_id": 104, "nombre": "Ana Pérez"}]}


async def test_buscar_cliente_accepts_partial_names():
    result = await buscar_cliente.ainvoke({"nombre": "Lucía"})
    assert result["clientes"][0]["cliente_id"] == 102


async def test_buscar_cliente_unknown_name_is_an_error_not_an_exception():
    result = await buscar_cliente.ainvoke({"nombre": "Zoe"})
    assert "error" in result


async def test_buscar_pedidos_totals_for_client_102():
    result = await buscar_pedidos.ainvoke({"cliente_id": 102})
    assert result["cantidad_pedidos"] == 3
    assert result["total"] == 14500
    assert [p["pedido_id"] for p in result["pedidos"]] == ["P-2001", "P-2002", "P-2003"]


async def test_buscar_pedidos_for_client_without_orders():
    result = await buscar_pedidos.ainvoke({"cliente_id": 103})
    assert result["cantidad_pedidos"] == 0
    assert result["total"] == 0


async def test_buscar_pedidos_unknown_client_returns_an_error():
    result = await buscar_pedidos.ainvoke({"cliente_id": 999})
    assert result == {"error": "no existe el cliente 999"}


async def test_detalle_pedido_returns_items_and_status():
    result = await detalle_pedido.ainvoke({"pedido_id": "P-2003"})
    assert result["estado"] == "en camino"
    assert result["total"] == 5500
    assert len(result["items"]) == 2


async def test_detalle_pedido_is_case_insensitive():
    result = await detalle_pedido.ainvoke({"pedido_id": "p-2001"})
    assert result["pedido_id"] == "P-2001"


async def test_detalle_pedido_unknown_order_returns_an_error():
    result = await detalle_pedido.ainvoke({"pedido_id": "P-0"})
    assert "error" in result


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_docstrings_are_descriptive(tool):
    # El modelo elige la herramienta leyendo solo esto.
    assert len(tool.description) > 150
    assert "Usala" in tool.description
