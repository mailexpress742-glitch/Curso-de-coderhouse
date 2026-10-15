import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from clients import errors
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
    """Interfaz común a todos los proveedores.

    Las subclases sólo saben hablar con su SDK. Los timeouts, los reintentos y
    la conversión de excepciones en errores controlados se resuelven acá.
    """

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

    @abstractmethod
    async def _generate(self, messages: Messages, params: GenerationParams) -> ModelResponse: ...

    @abstractmethod
    def _stream(
        self, messages: Messages, params: GenerationParams
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    def _classify_error(self, exc: Exception) -> LLMError: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def generate(
        self, messages: Messages, params: GenerationParams | None = None
    ) -> ModelResponse:
        params = params or GenerationParams()
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(params.timeout_s):
                    response = await self._generate(messages, params)
                break
            except Exception as exc:
                error = self._as_llm_error(exc, attempt)
                if not self._should_retry(error, attempt, params):
                    response = ModelResponse(provider=self.provider, model=self.model, error=error)
                    break
                await asyncio.sleep(self._retry_delay(exc, attempt))

        latency_ms = (time.perf_counter() - start) * 1000
        return response.model_copy(update={"latency_ms": latency_ms, "attempts": attempt})

    async def stream(
        self, messages: Messages, params: GenerationParams | None = None
    ) -> AsyncIterator[StreamChunk]:
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
                error = self._as_llm_error(exc, attempt)
                # Si ya se mostró texto, reintentar duplicaría la respuesta.
                if started or not self._should_retry(error, attempt, params):
                    yield StreamChunk(done=True, error=error)
                    return
                await asyncio.sleep(self._retry_delay(exc, attempt))

    def _as_llm_error(self, exc: Exception, attempt: int) -> LLMError:
        # asyncio.timeout() lanza TimeoutError, que no es una excepción del SDK.
        error = errors.timeout() if isinstance(exc, TimeoutError) else self._classify_error(exc)
        # Un error desconocido suele ser un bug: se loguea con el traceback.
        traceback = exc if error.kind == ErrorKind.UNKNOWN else None
        logger.warning(
            "%s, intento %d: %s (%s)",
            self.provider,
            attempt,
            error.message,
            error.kind,
            exc_info=traceback,
        )
        return error

    @staticmethod
    def _should_retry(error: LLMError, attempt: int, params: GenerationParams) -> bool:
        return error.retryable and attempt <= params.max_retries

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = _retry_after(exc)
        if retry_after is not None:
            return min(retry_after, self.retry_max_delay)
        backoff = min(self.retry_base_delay * 2 ** (attempt - 1), self.retry_max_delay)
        return backoff * random.uniform(0.5, 1)


def _retry_after(exc: Exception) -> float | None:
    """Segundos que pide esperar la API en el header retry-after, si lo mandó."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None
