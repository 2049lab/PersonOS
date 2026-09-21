"""Structured logs written to disk + a trace id threaded through one request.

We inject trace_id with loguru's contextualize rather than a patcher — the latter
loses context under enqueue (we got burned by this on an earlier project).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

_configured = False
_reinstalled = False   # Re-adding the file sink happens once (the platform logging
                       # library tears sinks down only once, at import time)

_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <5}</level> "
    "| <cyan>{extra[trace_id]}</cyan> | <level>{message}</level>"
)


def _file_sink(log_dir: Path) -> dict:
    """Config for the daily-rotating file sink; setup and the later re-add share this
    one definition so their behavior can't drift apart."""
    return dict(sink=log_dir / "personos_{time:YYYY-MM-DD}.log",
                format=_FMT, level="DEBUG",
                rotation="00:00", retention="14 days", encoding="utf-8")


def _redinfra_ready() -> bool:
    """Has the platform logging library already initialized logging (i.e. installed its
    own stdout tracing sink)? Not installed / unavailable -> False."""
    try:
        import redinfra.log.logger as _rl
        return bool(getattr(_rl, "_initialized", False))
    except Exception:   # noqa: BLE001
        return False


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Idempotent init: console + daily-rotating file sink, with trace_id in the format.

    The key point: if the platform logging library already installed its stdout tracing
    sink before this function runs (in some pods the startup order auto-inits it before
    the app), **never tear it down with logger.remove()** — otherwise application logs
    inside the container never reach the tracing backend again (that library's init is
    idempotent and will not reinstall). In that case we only add the file sink and
    leave stdout to it.
    """
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(extra={"trace_id": "-"})  # Default trace_id, so the file sink format never raises KeyError
    if _redinfra_ready():
        logger.add(**_file_sink(log_dir))       # Keep the platform's stdout sink, only add on-disk logging
    else:
        logger.remove()                         # Local / normal order: build our own stderr+file; if the platform
                                                # library runs later it will reconfigure stdout itself
        logger.add(sys.stderr, format=_FMT, level=level)
        logger.add(**_file_sink(log_dir))
    _configured = True


def _xray_patcher(record: dict) -> None:
    """Global loguru patcher: first run the platform's inject_context (which fills in
    userId and the tracing correlation fields), then overwrite the trace id with this
    request's trace_id — so all logs of one request share a single id across the
    ingest/recall worker threads, the log backend can stitch the request together, and
    the id matches the langfuse trace so you can jump between the two. Never raises."""
    try:
        from redinfra.log.context_injector import inject_context
        inject_context(record)
    except Exception:   # noqa: BLE001  Platform library unavailable: at least make sure these keys exist
        for k in ("xrayTraceId", "catRootId", "catParentId", "catMsgId", "userId"):
            record["extra"].setdefault(k, "")
    try:
        from . import obs
        tid = obs.request_trace_id()
        if tid:
            record["extra"]["xrayTraceId"] = tid
    except Exception:   # noqa: BLE001
        pass


def reinstall_file_sink(log_dir: Path) -> None:
    """After wiring up Redis, re-add the file sink and install the patcher that carries
    the request trace_id (idempotent, done only once).

    The platform's Redis pool module runs init_logger() at import time, which calls
    logger.remove() — tearing down every loguru sink in the process — and sets the
    patcher to its own inject_context (whose trace id stays empty, because we don't go
    through its RPC layer). Call this function after creating the pool to: (1) re-add
    the file sink so logs hit disk again; (2) swap the patcher for _xray_patcher so
    every log line carries this request's trace_id. Inside the container the platform's
    stdout sink still does the collecting, so both coexist.
    """
    global _reinstalled
    if _reinstalled or not _configured:
        return
    _reinstalled = True
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(**_file_sink(log_dir))
    logger.configure(patcher=_xray_patcher)     # Replace the platform's patcher (handlers untouched)


@contextmanager
def trace(trace_id: str):
    """Bind one request's trace_id to every log line emitted inside this scope."""
    with logger.contextualize(trace_id=trace_id):
        yield
