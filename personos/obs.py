"""Langfuse LLM observability instrumentation: a thin wrapper that becomes a complete
no-op with zero side effects when disabled.

Design rule: no pk/sk configured -> the client is None and every span/observation
context manager degrades to a no-op (nothing reported, main flow untouched).
**Instrumentation only observes; it must never change a business return value or an
exception path** — a failure in init, span creation, update, or flush is swallowed and
degraded, and the main path carries on.

Usage:
    with obs.root_span("recall", user_id=u, session_id=s, input=q) as span:
        with obs.observation("chat", model=m, input=msgs) as gen:
            resp = llm.chat(...)
            obs.update(gen, output=resp, usage={"input": pt, "output": ct, "total": tt})
"""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Optional

from loguru import logger

from .config import settings

_lock = threading.Lock()
_inited = False
_client = None   # Langfuse | None (created lazily)

# Name of the current LLM stage (R0/R5/R3'/W1/W2...), used by the model client to name
# its generation spans. The contextvar propagates along the call stack of one thread:
# stage() sets it at the entrance of a stage, and the chat call further down reads it.
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("lf_stage", default="")


@contextmanager
def stage(name: str):
    """Mark the current LLM stage (a nested call temporarily overrides it and restores
    the previous value on exit)."""
    tok = _stage.set(name or "")
    try:
        yield
    finally:
        _stage.reset(tok)


def current_stage() -> str:
    return _stage.get()


# trace_id of the current request (same id as the langfuse trace). The log patcher
# writes it into the tracing id field so that all logs of one request — across the
# ingest/recall worker threads — are stitched together. The contextvar propagates into
# child threads via copy_context.
_req_trace: contextvars.ContextVar[str] = contextvars.ContextVar("req_trace", default="")


def request_trace_id() -> str:
    return _req_trace.get()


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """A thread pool that preserves contextvars: submit() captures the submitting
    thread's context (including the active OTel span and the stage) and the worker runs
    the callable through copy_context().run — so LLM spans of parallel subtasks nest
    correctly under the parent trace and the stage propagates with them. The stdlib
    ThreadPoolExecutor drops the submitting thread's contextvars, which turns parallel
    LLM calls into orphan root traces (as seen in the weave-style writes). The
    platform's helper only has an asyncio version (async_to_sync), which doesn't apply
    to a synchronous fan-out."""

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)


def client():
    """Lazily build the langfuse client singleton; no pk/sk configured -> None (all
    instrumentation off). A failed init also degrades to None."""
    global _inited, _client
    if _inited:
        return _client
    with _lock:
        if _inited:
            return _client
        _inited = True
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.info("langfuse not configured (no pk/sk), LLM instrumentation disabled")
            return _client
        try:
            from langfuse import Langfuse
            _client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                environment=settings.langfuse_environment or None,
                release=settings.langfuse_release or None,
            )
            logger.info(f"langfuse enabled env={settings.langfuse_environment} host={settings.langfuse_host}")
        except Exception as e:   # noqa: BLE001  A failed instrumentation init must never take the service down
            logger.warning(f"langfuse init failed, instrumentation disabled: {e}")
            _client = None
    return _client


def _set_attrs(span, user_id, session_id, *, as_root: bool) -> None:
    """Mark as_root (required at the entry of a trace when the upstream isn't wired to
    langfuse) and attach user/session. Failures are ignored."""
    try:
        if as_root:
            span._otel_span.set_attribute("langfuse.internal.as_root", "true")   # must be a string
        if user_id:
            span._otel_span.set_attribute("langfuse.user.id", str(user_id))
        if session_id:
            span._otel_span.set_attribute("langfuse.session.id", str(session_id))
    except Exception:   # noqa: BLE001
        pass


@contextmanager
def root_span(name: str, *, user_id=None, session_id=None, input: Any = None,
              metadata: Any = None, trace_id: Optional[str] = None):
    """Root span at a request entry point (recall/ingest/profile/deep track). Marked
    as_root. Yields None when disabled or on failure.

    We reset the OTel context to empty before opening the span: when a thread pool
    (ingest_exec / FastAPI's anyio pool) reuses a thread, span context left over from
    the previous task bleeds into the next one, so its nested observations lose their
    parent and get thrown out as orphan root traces. Starting from an empty context,
    this task's attach/detach form their own stack — which both guarantees we really
    are the root and clears the leftovers, for recall, ingest and profile alike.

    trace_id (optional, 32 hex): threaded end to end — generated at the entry point and
    passed through the queue. If given, the langfuse trace uses that id and so does the
    tracing id in the logs (it is written into _req_trace even when langfuse is disabled
    or span creation fails, so the logs stay searchable).
    """
    c = client()

    @contextmanager
    def _bind(tid: str):
        """Bind the request trace_id to _req_trace (so log lines emitted anywhere in the request can carry it into
        the log tracing id); an empty value binds nothing."""
        tok = None
        try:
            if tid:
                tok = _req_trace.set(tid)
            yield
        finally:
            if tok is not None:
                try:
                    _req_trace.reset(tok)
                except Exception:   # noqa: BLE001
                    pass

    if c is None:                                  # langfuse off: still set trace_id so the logs carry it
        with _bind(trace_id or ""):
            yield None
        return
    reset_token = None
    try:
        from opentelemetry import context as _otel_ctx
        reset_token = _otel_ctx.attach(_otel_ctx.Context())   # Clean empty context, isolates thread leftovers
    except Exception:   # noqa: BLE001  OTel unavailable: degrade by not resetting, behavior as before
        reset_token = None
    try:
        kw = {"name": name, "input": input, "metadata": metadata}
        if trace_id:
            kw["trace_context"] = {"trace_id": trace_id}   # make the langfuse trace use the id passed in
        cm = c.start_as_current_span(**kw)
    except Exception as e:   # noqa: BLE001
        logger.debug(f"langfuse root_span failed (ignored): {e}")
        if reset_token is not None:
            try:
                from opentelemetry import context as _otel_ctx
                _otel_ctx.detach(reset_token)
            except Exception:   # noqa: BLE001
                pass
        with _bind(trace_id or ""):
            yield None
        return
    try:
        with cm as span:
            _set_attrs(span, user_id, session_id, as_root=True)
            with _bind(trace_id or current_trace_id() or ""):
                yield span
    finally:
        if reset_token is not None:
            try:
                from opentelemetry import context as _otel_ctx
                _otel_ctx.detach(reset_token)
            except Exception:   # noqa: BLE001
                pass


@contextmanager
def observation(name: str, *, as_type: str = "generation", model: Optional[str] = None,
                input: Any = None, metadata: Any = None):
    """One LLM/tool/retrieval observation, nested under the current span. Yields None
    when disabled or on failure."""
    c = client()
    if c is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(as_type=as_type, name=name, model=model,
                                            input=input, metadata=metadata)
    except Exception as e:   # noqa: BLE001
        logger.debug(f"langfuse observation failed (ignored): {e}")
        yield None
        return
    with cm as obs:
        yield obs


def update(obs, *, output: Any = None, usage: Optional[dict] = None,
           level: Optional[str] = None, status_message: Optional[str] = None) -> None:
    """Fill in an observation's output / token usage (usage values must be numbers) /
    error. A no-op when obs is None. Never raises."""
    if obs is None:
        return
    try:
        kw: dict = {}
        if output is not None:
            kw["output"] = output
        if usage:
            kw["usage_details"] = usage
        if level:
            kw["level"] = level
        if status_message:
            kw["status_message"] = status_message
        if kw:
            obs.update(**kw)
    except Exception:   # noqa: BLE001
        pass


def current_trace_id() -> Optional[str]:
    """Id of the currently active trace (for logs, or to return to the caller so they
    can look it up in the tracing backend). Disabled / nothing active -> None."""
    c = client()
    if c is None:
        return None
    try:
        return c.get_current_trace_id()
    except Exception:   # noqa: BLE001
        return None


def flush() -> None:
    """Flush buffered traces before exit (must be called by short-lived processes and
    at the end of a request)."""
    c = client()
    if c is not None:
        try:
            c.flush()
        except Exception:   # noqa: BLE001
            pass
