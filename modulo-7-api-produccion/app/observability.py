"""Observabilidad con Arize Phoenix y OpenInference.

`LangChainInstrumentor` instrumenta LangChain y LangGraph: cada ejecución del grafo
genera una traza con un span por nodo y por llamada al modelo, con los tokens de entrada y
salida. Phoenix calcula el costo a partir de esos tokens y del modelo usado.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from openinference.instrumentation import using_attributes
from opentelemetry import trace

from app.config import Settings

logger = logging.getLogger(__name__)

TRACER_NAME = "orquestador-api"


def setup_observability(settings: Settings) -> bool:
    """Conecta la app a Phoenix. Devuelve False si no hay un endpoint configurado."""
    if not settings.phoenix_endpoint:
        logger.info("PHOENIX_COLLECTOR_ENDPOINT no está definido: no se envían trazas")
        return False

    # El SDK de Anthropic también emite sus propios spans por cada llamada al modelo. Con
    # LangChain instrumentado, la misma llamada quedaría registrada dos veces y Phoenix
    # duplicaría los tokens y el costo. Hay que apagarlo antes de crear el cliente del modelo.
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    tracer_provider = register(
        project_name=settings.phoenix_project,
        endpoint=settings.phoenix_endpoint,
        batch=True,
        verbose=False,
    )
    # Solo se instrumenta LangChain/LangGraph (nodos y llamadas al modelo con sus tokens). No se
    # usa auto_instrument porque también trazaría cada request HTTP de FastAPI, y el polling del
    # cliente llenaría el dashboard de trazas irrelevantes que distorsionan la latencia.
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info(
        "trazas enviadas a %s (proyecto %s)", settings.phoenix_endpoint, settings.phoenix_project
    )
    return True


@contextmanager
def traced_job(job_id: str, kind: str) -> Iterator[None]:
    """Agrupa todo lo que ocurre en un job bajo una sesión y un span propios.

    `session_id` permite ver en Phoenix todas las trazas de un mismo job juntas; el span
    `job.<kind>` mide la duración total de la ejecución en el worker.
    """
    tracer = trace.get_tracer(TRACER_NAME)
    with (
        using_attributes(session_id=job_id, metadata={"job_id": job_id, "ejecucion": kind}),
        tracer.start_as_current_span(f"job.{kind}") as span,
    ):
        span.set_attribute("job.id", job_id)
        yield
