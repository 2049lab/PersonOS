"""The session write lock: ingest and session-end on the same (user, session) are
mutually exclusive, across replicas.

This used to be an in-process threading.Lock, which meant that with several
replicas, concurrent feeds into one session interleaved — two replicas could even
each build their own cell for the same segment. Now that it is external:

- RedisSessionLock uses SET NX PX plus a holder token, releasing through a Lua
  script that compares the token so a holder only ever deletes its own lock.
  Avoiding deadlock is valued above perfect mutual exclusion: once the TTL passes
  the lock lapses on its own and another replica may enter.
- MemorySessionLock is the in-process implementation for local and single-replica
  use, with a capped registry so it cannot leak.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, Protocol

from .redis_client import key as _key

# The hold TTL has to cover the worst case of a feed that is slow but not dead. A
# single chat call can already retry three times at a 120s timeout, about 365s,
# against an upstream that has slowed down without failing, plus headroom for
# building the cell afterwards. If the lock expires first, another replica walks
# into the same session and risks closing it twice. The price of the TTL is that a
# crashed holder leaves the session locked for up to that long — we accept it,
# because avoiding deadlock comes first.
LOCK_TTL_S = 600
# Upper bound on waiting for the lock. On timeout we raise TimeoutError so the task
# honestly reports "session busy" rather than queueing forever.
LOCK_WAIT_S = 60
_RETRY_S = 0.2

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# Renewal only extends the lock if we still hold it (the token matches), so we
# cannot accidentally renew somebody else's. A single EVAL, like release.
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class SessionLock(Protocol):
    def __call__(self, user_id: str, session_id: str): ...   # -> ContextManager[None]
    def try_acquire(self, user_id: str, session_id: str): ...  # -> handle | None (non-blocking)
    def release(self, user_id: str, session_id: str, handle) -> None: ...
    def renew(self, user_id: str, session_id: str, handle) -> bool: ...  # renew; succeeds only while still held


class RedisSessionLock:
    """A distributed lock in Redis: key = {env}:personos:lock:{user}:{session}."""

    def __init__(self, client, ttl_s: int = LOCK_TTL_S, wait_s: float = LOCK_WAIT_S):
        self._c = client
        self._ttl = ttl_s
        self._wait = wait_s

    @contextmanager
    def __call__(self, user_id: str, session_id: str) -> Iterator[None]:
        k = _key("lock", user_id, session_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self._wait
        while not self._c.set(k, token, nx=True, px=self._ttl * 1000):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting {self._wait}s for the session write lock: {user_id}/{session_id}")
            time.sleep(_RETRY_S)
        try:
            yield
        finally:
            self._c.eval(_RELEASE_LUA, 1, k, token)

    def try_acquire(self, user_id: str, session_id: str) -> str | None:
        """Try to take the lock without blocking.

        On success it returns the token, which release uses to delete only our own
        lock; on failure it returns None, meaning someone else holds it.

        This is for the dispatcher: if it cannot get the lock it returns at once and
        goes to work on another session, never spinning and never tying up a thread.
        """
        k = _key("lock", user_id, session_id)
        token = uuid.uuid4().hex
        return token if self._c.set(k, token, nx=True, px=self._ttl * 1000) else None

    def release(self, user_id: str, session_id: str, handle) -> None:
        """Release using the token; the Lua script compares it so we delete only our own lock. An empty token is a no-op."""
        if not handle:
            return
        self._c.eval(_RELEASE_LUA, 1, _key("lock", user_id, session_id), handle)

    def renew(self, user_id: str, session_id: str, handle) -> bool:
        """Extend the lock's TTL, succeeding only while we still hold it.

        A long drain calls this once per item it processes, so the lock cannot
        expire mid-way and let another replica in.
        """
        if not handle:
            return False
        r = self._c.eval(_RENEW_LUA, 1, _key("lock", user_id, session_id), handle, self._ttl * 1000)
        return bool(r)


class MemorySessionLock:
    """An in-process lock registry, for the single-replica shape.

    It is capped: once over the limit, lock objects nobody currently holds are
    cleared out.
    """

    _CAP = 4096

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}   # a tuple key, so segments containing "::" still cannot collide

    @contextmanager
    def __call__(self, user_id: str, session_id: str) -> Iterator[None]:
        k = (user_id, session_id)
        with self._guard:
            lk = self._locks.setdefault(k, threading.Lock())
            if len(self._locks) > self._CAP:     # over the cap: clear old unheld locks, excluding the one just taken
                for stale in [s for s, lock in self._locks.items() if s != k and not lock.locked()]:
                    self._locks.pop(stale, None)
        with lk:
            yield

    def _lock_for(self, k):
        with self._guard:
            lk = self._locks.setdefault(k, threading.Lock())
            if len(self._locks) > self._CAP:     # over the cap: clear old unheld locks, so the try_acquire path cannot leak either
                for stale in [s for s, lock in self._locks.items() if s != k and not lock.locked()]:
                    self._locks.pop(stale, None)
            return lk

    def try_acquire(self, user_id: str, session_id: str):
        """Try to take the lock without blocking: the Lock handle for release on success, None on failure."""
        lk = self._lock_for((user_id, session_id))
        return lk if lk.acquire(blocking=False) else None

    def release(self, user_id: str, session_id: str, handle) -> None:
        if handle is not None:
            handle.release()

    def renew(self, user_id: str, session_id: str, handle) -> bool:
        return handle is not None      # an in-process lock never expires, so holding it is already a successful renewal
