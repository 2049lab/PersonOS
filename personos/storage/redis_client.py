"""Redis access, used only when several workers share session state.

Optional: without PERSONOS_REDIS_URL the whole module stays unused and the
in-memory implementations take over. Requires ``pip install personos[redis]``.

- Lazy: importing this module touches no network. The client is built on the
  first get_redis() call, so a process that never uses Redis never connects.
- Every key is built through key(), as "{env}:{app}:{remaining segments}". The
  environment comes from PERSONOS_ENV and defaults to "local", so two
  deployments pointed at one Redis cannot collide, and keys stay greppable and
  removable by segment.
- Every key must carry a TTL. A shared Redis must not accumulate keys that
  nothing will ever delete.
"""

from __future__ import annotations

import os
import threading

from ..config import settings

_lock = threading.Lock()
_client = None   # lazy singleton; one connection pool shared per process, whose instance list a background thread refreshes


def get_redis():
    """The Redis client singleton, connected on first use.

    Raises if PERSONOS_REDIS_URL is unset: reaching this function at all means
    something asked for shared state, and quietly handing back a single-process
    stand-in would turn a configuration mistake into silent data divergence
    between workers.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not settings.redis_url:
                    from ..errors import MissingCapability

                    raise MissingCapability(
                        "shared session state",
                        "no Redis URL is configured",
                        "set PERSONOS_REDIS_URL (needed only when several workers "
                        "share one memory store)")
                try:
                    import redis as _redis
                except ImportError as e:  # pragma: no cover - depends on extras
                    raise ImportError(
                        "Redis support needs the client: pip install 'personos[redis]'"
                    ) from e
                _client = _redis.Redis.from_url(settings.redis_url,
                                                decode_responses=False)
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
