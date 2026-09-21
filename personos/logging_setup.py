"""Structured logs on disk, with a trace id threaded through one request.

The trace id is bound with loguru's ``contextualize`` rather than a patcher: a
patcher loses the binding when a sink runs with ``enqueue``, which produces logs
that look fine right up until the day you need to correlate them.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

_configured = False

_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <5}</level> "
    "| <cyan>{extra[trace_id]}</cyan> | <level>{message}</level>"
)


def _file_sink(log_dir: Path) -> dict:
    """Config for the daily-rotating file sink.

    Retention is bounded on purpose: an unbounded log directory eventually
    becomes the reason a process dies, and that cause is never where anyone
    looks first.
    """
    return dict(sink=log_dir / "personos_{time:YYYY-MM-DD}.log",
                format=_FMT, level="DEBUG",
                rotation="00:00", retention="14 days", encoding="utf-8")


def setup_logging(log_dir: Path | str, level: str = "INFO") -> None:
    """Install a stderr sink and a file sink. Idempotent.

    Called from Memory's constructor rather than at import time, so that merely
    importing the library neither creates directories nor rearranges the
    logging configuration of the application embedding it.
    """
    global _configured
    if _configured:
        return
    path = Path(log_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # noqa: BLE001
        # An unwritable log directory must not stop the library from working.
        # Logs are diagnostics, not the product.
        logger.warning(f"cannot create the log directory {path} ({e}); "
                       f"logging to stderr only")
        path = None

    logger.remove()
    # A default trace id, so the format string cannot raise KeyError on a log
    # line emitted outside any request scope.
    logger.configure(extra={"trace_id": "-"})
    logger.add(sys.stderr, format=_FMT, level=level)
    if path is not None:
        logger.add(**_file_sink(path))
    _configured = True


@contextmanager
def trace(trace_id: str):
    """Bind one request's trace id to every log line emitted inside this scope.

    Worth the ceremony: the ingest and recall paths hand work to thread pools,
    so without this the lines belonging to one request are interleaved with
    everyone else's and cannot be told apart afterwards.
    """
    with logger.contextualize(trace_id=trace_id):
        yield
