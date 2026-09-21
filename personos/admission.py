"""Cap on in-flight background tasks: reject (503) when full, release on completion.

When upstreams fail or tasks generally slow down, accepting 202s without limit lets
the thread-pool queue grow unbounded (every entry holds on to a request payload and
a closure) — this is the last line of defense against memory slowly being eaten up.
A small standalone module with no import side effects, so unit tests can import it
directly (no runtime singleton, no DB connection).
"""
from __future__ import annotations

import threading

_MAX_PENDING = 200   # Max in-flight background tasks (queued + running)


class TaskOverloaded(RuntimeError):
    """In-flight capacity is full: callers should fail fast / retry later instead of
    piling on more work."""


class AdmissionGate:
    """In-flight permit gate (a wrapper around threading.BoundedSemaphore semantics):
    reject when full, release when the task reaches a terminal state.

    Release happens across threads (the submitting thread takes the permit, the worker
    releases it); releasing more times than acquired raises ValueError, which is
    exactly the self-check that "every path releases"."""

    def __init__(self, cap: int = _MAX_PENDING):
        self._sem = threading.BoundedSemaphore(cap)

    def try_enter(self) -> bool:
        return self._sem.acquire(blocking=False)

    def leave(self) -> None:
        self._sem.release()
