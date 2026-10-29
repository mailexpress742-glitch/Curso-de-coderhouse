import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from config import Settings
from schemas import ExtraccionTecnica, IncompleteOutputError, PipelineError, Provider

logger = logging.getLogger(__name__)

# Motivos de fin que indican que la respuesta quedó cortada (OpenAI / Anthropic).
TRUNCATED_REASONS = {"length", "max_tokens", "model_context_window_exceeded"}

# Errores de formato: tiene sentido volver a preguntar. Los errores de red y los
# rate limits ya los reintenta el SDK del proveedor, así que no se duplican acá.
RETRYABLE_ERRORS = (OutputParserException, ValidationError, IncompleteOutputError)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un analista técnico. Extraés información estructurada de textos sobre "
            "software: descripciones de arquitectura, logs de error, incidentes.\n"
            "- tecnologias: solo las que aparecen en el texto, sin inventar ninguna. Si no "
            'aparece ninguna, devolvé ["no identificada"].\n'
            "- nivel_de_criticidad: alta si hay caída de servicio, pérdida de datos o riesgo de "
            "seguridad; media si hay degradación o riesgo a futuro; baja si es informativo.\n"
            "- resumen_tecnico: una o dos oraciones, en español.",
        ),
        ("human", "Texto a analizar:\n\n{text}"),
    ]
)


def build_model(settings: Settings) -> BaseChatModel:
    """Crea el modelo de LangChain para el proveedor configurado (igual que en el módulo 1)."""
    key = settings.api_key.get_secret_value()
    if settings.provider == Provider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model, api_key=key)

    from langchain_anthropic import ChatAnthropic

    # Estos modelos no aceptan temperature, por eso no se envía.
    return ChatAnthropic(model=settings.model, api_key=key, max_tokens=2_000)


def check_output(result: dict[str, Any]) -> ExtraccionTecnica:
    """Revisa la salida cruda del modelo y devuelve el objeto validado o lanza un error.

    Con include_raw=True LangChain no lanza excepciones por JSON malformado: las
    deja en result['parsing_error']. Las levantamos acá para que with_retry las vea.
    """
    raw = result["raw"]
    metadata = getattr(raw, "response_metadata", {}) or {}
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")

    if reason in TRUNCATED_REASONS:
        logger.warning("respuesta cortada por el modelo (%s), se va a reintentar", reason)
        raise IncompleteOutputError(f"salida incompleta ({reason})")

    if result["parsing_error"] is not None:
        logger.warning("la salida no cumple el esquema: %s", result["parsing_error"])
        raise result["parsing_error"]

    if result["parsed"] is None:
        logger.warning("el modelo no devolvió el esquema pedido")
        raise IncompleteOutputError("el modelo no devolvió el esquema pedido")

    logger.info("salida validada (finish_reason=%s)", reason)
    return result["parsed"]


def build_chain(
    model: BaseChatModel, *, max_attempts: int = 3, jitter: bool = True
) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """prompt | modelo con salida estructurada | validación, con reintentos.

    El reintento envuelve al modelo y a la validación juntos: si la salida es
    inválida, se vuelve a llamar al modelo.
    """
    structured_model = model.with_structured_output(
        ExtraccionTecnica, method="json_schema", include_raw=True
    )
    validated_model = (structured_model | RunnableLambda(check_output)).with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=jitter,
    )
    return PROMPT | validated_model


async def process_text(
    text: str, chain: Runnable[dict[str, str], ExtraccionTecnica]
) -> ExtraccionTecnica:
    """Extrae las entidades técnicas de un texto.

    Lanza PipelineError si no logra una salida válida después de los reintentos.
    """
    if not text.strip():
        raise ValueError("el texto a procesar está vacío")

    logger.info("procesando texto de %d caracteres", len(text))
    try:
        return await chain.ainvoke({"text": text})
    except RETRYABLE_ERRORS as exc:
        logger.error("se agotaron los reintentos: %s", exc)
        raise PipelineError("no se obtuvo una extracción válida después de los reintentos") from exc
