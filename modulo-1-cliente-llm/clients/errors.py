from schemas import ErrorKind, LLMError


def from_status(status_code: int, message: str) -> LLMError:
    if status_code in (401, 403):
        kind = ErrorKind.AUTHENTICATION
    elif status_code == 429:
        kind = ErrorKind.RATE_LIMIT
    elif status_code >= 500 or status_code in (408, 409):
        kind = ErrorKind.SERVER
    elif status_code >= 400:
        kind = ErrorKind.INVALID_REQUEST
    else:
        kind = ErrorKind.UNKNOWN

    return LLMError(
        kind=kind,
        message=message,
        retryable=kind in (ErrorKind.RATE_LIMIT, ErrorKind.SERVER),
        status_code=status_code,
    )


def timeout() -> LLMError:
    return LLMError(
        kind=ErrorKind.TIMEOUT,
        message="el modelo no respondió a tiempo",
        retryable=True,
    )


def connection(exc: Exception) -> LLMError:
    return LLMError(kind=ErrorKind.CONNECTION, message=str(exc), retryable=True)


def unknown(exc: Exception) -> LLMError:
    return LLMError(kind=ErrorKind.UNKNOWN, message=f"{type(exc).__name__}: {exc}")


def refusal(detail: str) -> LLMError:
    return LLMError(kind=ErrorKind.REFUSAL, message=f"el modelo no quiso responder ({detail})")
