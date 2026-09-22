"""PersonOS FastAPI entry point: mounts only the public memory service (service_api,
prefixed /api/v1).

Run with:  uvicorn server.app:app --reload --port 8000
           python -m personos      # production, single process (host/port from env)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from personos.config import settings
from personos.storage.redis_client import get_redis
from server.runtime import rt
from personos.online.event_forward import install as _install_event_forward
from .response import install as _install_envelope
from .api import router as service_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background ingest dispatcher in-process (it consumes the session
    queues); stop it gracefully on exit.

    One dispatcher per pod scans the board and hands sessions with pending work to the
    ingest pool for FIFO consumption — across replicas the session lock keeps it
    single-flight, so no session is consumed twice.
    """
    rt.start_dispatcher()
    logger.info("PersonOS started: ingest dispatcher ready")
    try:
        yield
    finally:
        rt.stop_dispatcher()
        # Graceful shutdown: wait for in-flight drain jobs to finish (they hold session
        # locks and only release them when done) and cancel queued jobs that haven't
        # started. Without waiting, a hard kill leaves in-flight session locks held until
        # their TTL expires (no message is lost — the processing set recovers reliably).
        rt.ingest_exec.shutdown(wait=True, cancel_futures=True)
        # The profile pool winds down the same way: an in-flight consolidation holds the
        # per-user single-flight lock until it finishes; queued ones are cancelled (the
        # next trigger heals it).
        rt.profile_exec.shutdown(wait=True, cancel_futures=True)
        # The video pool likewise: in-flight identity jobs finish, queued ones drop
        # (their clips stay queued and the next dispatcher round picks them up).
        rt.video_exec.shutdown(wait=True, cancel_futures=True)
        from personos import obs
        obs.flush()                                    # Flush buffered langfuse traces
        logger.info("PersonOS stopped: ingest dispatcher wound down")


app = FastAPI(title="PersonOS", lifespan=lifespan)


# -- Static API documentation page: generated at build time by
# scripts/build_api_doc.py and served directly at runtime with no dependencies --
_API_DOC_HTML = ""
try:
    _API_DOC_HTML = (Path(__file__).parent / "api_doc.html").read_text(encoding="utf-8")
except OSError:
    logger.warning("api_doc.html is missing (not built? run scripts/build_api_doc.py); "
                   "/ and /api-doc will return a placeholder")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api-doc", response_class=HTMLResponse, include_in_schema=False)
def api_doc():
    """API documentation page: callers and agents can read the latest usage by hitting
    the root path of this domain."""
    return _API_DOC_HTML or "<h1>personos-mem</h1><p>API docs not built (server/api_doc.html missing)</p>"


# -- Platform probes: outside /api/v1, unauthenticated --
@app.get("/healthz")
async def healthz():
    """Liveness probe: ok as long as the process can respond; touches neither the DB nor
    any external dependency. Async on purpose: it runs on the event loop, so it stays
    responsive even when all 40 anyio worker threads are occupied by slow requests
    (a liveness timeout would make K8s restart the pod — the worst outcome under load)."""
    return {"status": "ok"}


# Stays a sync def: it issues a blocking DB ping, which must not run on the event loop.
# Queuing behind a saturated thread pool is acceptable here — under overload, readiness
# flapping (pod leaves rotation) is the desired backpressure, unlike a liveness restart.
@app.get("/readyz")
def readyz():
    """Readiness probe: only take traffic once MySQL and (if enabled) Redis are
    reachable — they are hard dependencies of the write path."""
    try:
        rt.db.fetch_one("SELECT 1")
        if settings.redis_url:
            get_redis().ping()
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"status": "not-ready", "error": str(e)})


app.include_router(service_router)                 # The public memory service: the only router
_install_envelope(app)                             # Global exception handlers: exceptions raised by the framework or dependencies also get the envelope (only under /api/v1)

# Memory-change event forwarding (optional): with MEMORY_EVENT_URL unset no callback is
# installed at all, so there is zero overhead.
# The memory service only broadcasts "what changed"; subscription, signing and delivery
# all live in the application layer.
_install_event_forward()
