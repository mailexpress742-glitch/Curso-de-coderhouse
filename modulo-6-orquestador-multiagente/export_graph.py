"""Genera el diagrama Mermaid del grafo en docs/grafo.mmd (el mismo que está en el README).

python export_graph.py
"""

from langchain_core.language_models import FakeListChatModel

from config import BASE_DIR
from graph import Models, build_graph


class _Placeholder(FakeListChatModel):
    """El diagrama no llama a ningún modelo, así que no hace falta una API key."""

    def bind_tools(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self

    def with_structured_output(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self


def main() -> None:
    graph = build_graph(Models.same(_Placeholder(responses=["x"])))
    mermaid = graph.get_graph().draw_mermaid()
    target = BASE_DIR / "docs" / "grafo.mmd"
    target.write_text(mermaid, encoding="utf-8")
    print(mermaid)
    print(f"\nGuardado en {target}")


if __name__ == "__main__":
    main()
