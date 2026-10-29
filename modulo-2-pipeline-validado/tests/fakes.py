from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from schemas import ExtraccionTecnica

VALID = ExtraccionTecnica(
    tecnologias=["FastAPI", "Redis"],
    nivel_de_criticidad="alta",
    resumen_tecnico="API en FastAPI con sesiones en Redis y límite de conexiones.",
)


def raw(finish_reason: str = "stop") -> AIMessage:
    return AIMessage(content="", response_metadata={"finish_reason": finish_reason})


def good() -> dict[str, Any]:
    return {"raw": raw(), "parsed": VALID, "parsing_error": None}


def malformed() -> dict[str, Any]:
    try:
        ExtraccionTecnica.model_validate({"tecnologias": [], "nivel_de_criticidad": "urgente"})
    except Exception as exc:
        return {"raw": raw(), "parsed": None, "parsing_error": exc}
    raise AssertionError("el esquema debería haber fallado")


def truncated() -> dict[str, Any]:
    return {"raw": raw("length"), "parsed": None, "parsing_error": None}


class FakeModel:
    """Simula un chat model: with_structured_output devuelve un runnable que va
    entregando las respuestas del guion, una por llamada."""

    def __init__(self, *script: dict[str, Any]) -> None:
        self.script = list(script)
        self.calls = 0

    def with_structured_output(
        self, schema: type, *, method: str, include_raw: bool
    ) -> RunnableLambda:
        assert schema is ExtraccionTecnica and include_raw

        def respond(_: Any) -> dict[str, Any]:
            result = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            return result

        return RunnableLambda(respond)
