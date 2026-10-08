"""Clase base abstracta de los clientes de LLM.

La lógica que no depende del proveedor vive acá: timeouts, reintentos con
backoff exponencial y conversión de excepciones en errores controlados. Cada
proveedor sólo implementa cómo se habla con su SDK (`_generate`, `_stream`) y
cómo se traducen sus excepciones (`_classify_error`).
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import (
    ErrorKind,
    GenerationParams,
    LLMError,
    Messages,
    ModelResponse,
    Provider,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    provider: Provider

    def __init__(
        self,
        model: str,
        *,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ) -> None:
        self.model = model
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    # ------------------------------------------------------------------
    # Lo que implementa cada proveedor
    # ------------------------------------------------------------------

    @abstractmethod
    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse:
        """Una sola llamada al SDK, sin reintentos. Puede lanzar excepciones."""

    @abstractmethod
    def _stream(self, messages: Messages, params: GenerationParams) -> AsyncIterator[StreamChunk]:
        """Generador asíncrono sobre el stream del SDK. Termina con un chunk `done=True`."""

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a un `LLMError`."""

    async def aclose(self) -> None:
        """Libera las conexiones HTTP del SDK."""

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def generate(
        self, messages: Messages, params: GenerationParams | None = None
    ) -> ModelResponse:
        """Respuesta completa. Nunca lanza: los fallos vuelven en `ModelResponse.error`."""
        params = params or GenerationParams()
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(params.timeout_s):
                    response = await self._generate(messages, params)
            except Exception as exc:
                error = self._to_error(exc, attempt)
                if self._should_retry(error, attempt, params):
                    await asyncio.sleep(self._retry_delay(exc, attempt))
                    continue
                response = ModelResponse(provider=self.provider, model=self.model, error=error)
            latency_ms = (time.perf_counter() - start) * 1000
            return response.model_copy(update={"latency_ms": latency_ms, "attempts": attempt})

    async def stream(
        self, messages: Messages, params: GenerationParams | None = None
    ) -> AsyncIterator[StreamChunk]:
        """Streaming de tokens. Nunca lanza: un fallo llega como último chunk con `error`.

        Sólo se reintenta si el fallo ocurre antes del primer chunk; si ya se
        emitió texto, reintentar duplicaría la respuesta en pantalla.
        """
        params = params or GenerationParams()
        attempt = 0
        while True:
            attempt += 1
            started = False
            try:
                async for chunk in self._stream(messages, params):
                    started = True
                    yield chunk
                return
            except Exception as exc:
                error = self._to_error(exc, attempt)
                if not started and self._should_retry(error, attempt, params):
                    await asyncio.sleep(self._retry_delay(exc, attempt))
                    continue
                yield StreamChunk(done=True, error=error)
                return

    # ------------------------------------------------------------------
    # Manejo de errores y reintentos
    # ------------------------------------------------------------------

    def _to_error(self, exc: Exception, attempt: int) -> LLMError:
        if isinstance(exc, TimeoutError):  # lanzado por asyncio.timeout()
            error = LLMError(
                kind=ErrorKind.TIMEOUT,
                message="el modelo no respondió dentro del timeout",
                retryable=True,
            )
        else:
            error = self._classify_error(exc)
        logger.warning(
            "%s: intento %d falló (%s): %s", self.provider, attempt, error.kind, error.message
        )
        if error.kind is ErrorKind.UNKNOWN:
            logger.debug("traceback del error desconocido", exc_info=exc)
        return error

    @staticmethod
    def _should_retry(error: LLMError, attempt: int, params: GenerationParams) -> bool:
        return error.retryable and attempt <= params.max_retries

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Backoff exponencial con jitter. Si la API mandó `retry-after`, se respeta."""
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, self.retry_max_delay)
        backoff = self.retry_base_delay * 2 ** (attempt - 1)
        return min(backoff, self.retry_max_delay) * random.uniform(0.5, 1.0)


def error_from_status(status_code: int, message: str) -> LLMError:
    """Clasificación por código HTTP, para los errores que no tienen clase propia."""
    if status_code in (401, 403):
        kind, retryable = ErrorKind.AUTHENTICATION, False
    elif status_code == 429:
        kind, retryable = ErrorKind.RATE_LIMIT, True
    elif status_code in (408, 409):
        kind, retryable = ErrorKind.SERVER, True
    elif status_code >= 500:
        kind, retryable = ErrorKind.SERVER, True
    elif 400 <= status_code < 500:
        kind, retryable = ErrorKind.INVALID_REQUEST, False
    else:
        kind, retryable = ErrorKind.UNKNOWN, False
    return LLMError(kind=kind, message=message, retryable=retryable, status_code=status_code)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None
