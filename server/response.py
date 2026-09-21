"""A single response envelope {code, data, msg} for the public API, assembled centrally
so no endpoint has to be changed one by one.

The contract (agreed with callers):
- **Success**: the HTTP status follows REST semantics (200/201/202), and the body has
  `code=0`, `data=<payload>`, `msg="ok"`.
- **Failure**: the HTTP status carries the meaning (4xx/5xx), and the body has
  `code=<that HTTP status number>` (a coarse code in the same family as HTTP),
  `data=null`, `msg=<human-readable reason>`.

Two pieces do this centrally, with zero changes to any handler:
1. `EnvelopeRoute`: a custom APIRoute that wraps whatever each handler **returns** (a
   success dict, or the older style `JSONResponse({"error": ...})`) into the envelope,
   preserving the original status code and headers (such as Retry-After).
2. `install(app)`: registers global exception handlers that take over HTTPExceptions
   **raised** by the framework or dependencies (401/404...), request validation errors
   (422) and unexpected exceptions (500) — none of which pass through a handler's return
   value, so the route layer cannot catch them.

Both apply only to `/api/v1`; the platform probes (/healthz, /readyz) stay as they are,
since the orchestrator doesn't parse the envelope.
The shared constructors `ResponseUtils.ok/error` are used by internal code and the
exception handlers alike.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

_API_PREFIX = "/api/v1"
CODE_OK = 0
_ENVELOPE_KEYS = {"code", "data", "msg"}


def _envelope(code: int, data: Any, msg: str) -> dict:
    return {"code": code, "data": data, "msg": msg}


class ResponseUtils:
    """Shared response constructors: internal code and the exception handlers use one
    and the same envelope logic."""

    @staticmethod
    def ok(data: Any = None, msg: str = "ok", *, status_code: int = 200,
           headers: dict | None = None) -> JSONResponse:
        return JSONResponse(status_code=status_code, headers=headers,
                            content=_envelope(CODE_OK, data, msg))

    @staticmethod
    def error(http_status: int, msg: str, *, code: int | None = None,
              data: Any = None, headers: dict | None = None) -> JSONResponse:
        """The failure envelope. code defaults to the HTTP status number (a coarse code
        in the same family as HTTP); msg is the human-readable reason."""
        return JSONResponse(status_code=http_status, headers=headers,
                            content=_envelope(code if code is not None else http_status, data, msg))


def _is_enveloped(body: Any) -> bool:
    return isinstance(body, dict) and set(body.keys()) == _ENVELOPE_KEYS


def _err_msg(body: Any, fallback: str) -> str:
    """Pull a human-readable reason out of an older-style error body: prefer error /
    detail / msg, and use the fallback when none of them is there."""
    if isinstance(body, dict):
        for k in ("error", "detail", "msg", "message"):
            v = body.get(k)
            if isinstance(v, str) and v.strip():
                return v
    if isinstance(body, str) and body.strip():
        return body
    return fallback


def _wrap_response(resp) -> Any:
    """Wrap the response a handler returned into the envelope. Non-JSON, or already an
    envelope -> returned unchanged.

    The original status code and headers are preserved (rate-limit headers such as
    Retry-After must pass through); success uses code=0, failure uses the HTTP status
    number as the code.
    """
    body_bytes = getattr(resp, "body", None)
    media = (getattr(resp, "media_type", "") or "").lower()
    if body_bytes is None or ("json" not in media and media != ""):
        return resp                                  # Streaming / non-JSON: leave it alone
    try:
        body = json.loads(body_bytes) if body_bytes else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return resp                                  # Can't parse it: return as-is
    if _is_enveloped(body):
        return resp                                  # Already an envelope: don't wrap twice
    status = resp.status_code
    # Strip content-length/type so the new JSONResponse recomputes them; every other
    # header (Retry-After and friends) passes through
    headers = {k: v for k, v in resp.headers.items()
               if k.lower() not in ("content-length", "content-type")}
    if status < 400:
        return ResponseUtils.ok(data=body, status_code=status, headers=headers or None)
    return ResponseUtils.error(status, _err_msg(body, "request failed"), headers=headers or None)


class EnvelopeRoute(APIRoute):
    """A custom APIRoute: responses a handler returns normally are wrapped in the
    envelope (raised exceptions are handled by the global handlers)."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def custom(request: Request):
            return _wrap_response(await original(request))

        return custom


# -- Global exception handlers: they take over exceptions raised by the framework or
# dependencies, i.e. the paths a handler's return value can never cover --

def _http_exc_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=exc.status_code, content={"detail": detail},
                            headers=getattr(exc, "headers", None))   # Not a business endpoint: leave it as-is
    return ResponseUtils.error(exc.status_code, detail or "request failed",
                               headers=getattr(exc, "headers", None))


def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errs = exc.errors()
    first = errs[0] if errs else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    msg = f"request validation failed: {loc or '?'} {first.get('msg', '')}".strip()
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=422, content={"detail": errs})
    return ResponseUtils.error(422, msg)


def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"unhandled exception path={request.url.path}: {exc!r}")
    if not request.url.path.startswith(_API_PREFIX):
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    return ResponseUtils.error(500, "internal server error")   # Never leak the stack or internal details (they are already in the logs)


def install(app: FastAPI) -> None:
    """Register the three kinds of global exception handler on the app (only /api/v1
    returns the envelope; everything else keeps the default behavior)."""
    app.add_exception_handler(StarletteHTTPException, _http_exc_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
