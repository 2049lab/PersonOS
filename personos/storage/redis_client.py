"""Redis access through a service-discovery connection pool.

- Lazy: importing this module touches no network. The pool is built on the first
  get_redis() call, which is when service discovery resolves the cluster. A
  service or a test that never really uses Redis therefore needs none of the
  discovery environment variables.
- Every key is built through key(), as "{env}:{app}:{remaining segments}". The
  environment comes from PERSONOS_ENV, which is unset locally and so defaults to
  "local", keeping local keys naturally separate from deployed ones. The prefix is
  what makes keys distinguishable inside a shared cluster and lets them be cleaned
  up by segment.
- A constraint: every key must carry a TTL, since a shared cluster must not
  accumulate permanent keys.
- A dependency constraint: redis-py must stay pinned to 6.x, because the
  discovery pool library's handshake protocol is incompatible with 8.x. The pin
  lives in requirements.
"""

from __future__ import annotations

import os
import threading

from ..config import settings

_lock = threading.Lock()
_client = None   # lazy singleton; one connection pool shared per process, whose instance list a background thread refreshes


def get_redis():
    """Get the Redis client singleton.

    The first call builds the pool, resolving the cluster through service
    discovery. Failures are raised as they are.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import redis as _redis
                from redinfra.redis.pool import DiscoveryBlockingConnectionPool
                # The pool library calls init_logger() at import time, which calls
                # logger.remove() and tears out every sink we installed, leaving
                # only its own stdout sink. So re-attach the file sink immediately
                # after building the pool, or logs stop reaching disk.
                from ..logging_setup import reinstall_file_sink
                reinstall_file_sink(settings.log_dir)
                _client = _redis.Redis(
                    connection_pool=DiscoveryBlockingConnectionPool(
                        cluster_name=settings.redis_cluster))
    return _client


def _esc(part: str) -> str:
    """Escape within a segment: ":" becomes "%3A" and "%" becomes "%25".

    The order matters — % first, then : — so the second replacement cannot rewrite
    what the first produced.
    """
    return part.replace("%", "%25").replace(":", "%3A")


def key(*parts: str) -> str:
    """Build a normalized key: f"{env}:personos:{parts...}", e.g. dev:personos:seg:u1:s1.

    Each segment is separator-escaped before being joined, because segment content
    is caller-controlled (user_id, session_id). Without escaping, (u="a:b", s="c")
    and (u="a", s="b:c") produce the same key — which means one user reading and
    writing another's data, and the seg keys hold raw conversation text, the most
    sensitive thing here. There is a character whitelist at the entry layer
    (session_scope) as well; this is defence in depth, so that even a path that
    skips validation cannot construct a colliding key.
    """
    env = settings.env
    return f"{env}:personos:" + ":".join(_esc(p) for p in parts)
