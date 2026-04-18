"""Map upstream OpenAI SDK errors to HTTP responses with clear JSON bodies."""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse

from core.logging_config import get_logger

logger = get_logger(__name__)


def _hint(code: int | None, msg: str) -> str:
    m = msg.lower()
    if code == 429 or "quota" in m or "insufficient_quota" in m:
        return (
            "Quota or rate limit: add billing/credits at "
            "https://platform.openai.com/account/billing and check usage limits."
        )
    if code == 401:
        return "Invalid or missing API key — set OPENAI_API_KEY in `.env` at the project root and restart."
    return "See https://platform.openai.com/docs/guides/error-codes for API error codes."


def _openai_error_payload(exc: Any) -> tuple[int, dict[str, Any]]:
    code = getattr(exc, "status_code", None)
    msg = getattr(exc, "message", None) or str(exc)
    err_type = getattr(exc, "type", None) or "openai_error"

    if code == 401:
        http = status.HTTP_401_UNAUTHORIZED
    elif code == 429:
        http = status.HTTP_503_SERVICE_UNAVAILABLE
    elif code == 403:
        http = status.HTTP_403_FORBIDDEN
    else:
        http = status.HTTP_502_BAD_GATEWAY

    body = {
        "detail": "openai_api_error",
        "openai_status": code,
        "error_type": err_type,
        "message": msg,
        "hint": _hint(code, msg),
    }
    return http, body


async def openai_api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("openai_api_error", path=request.url.path, error=str(exc))
    http, body = _openai_error_payload(exc)
    return JSONResponse(status_code=http, content=body)
