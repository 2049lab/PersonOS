"""Access to unclosed segments — the write-path state — as session state shared
across requests and across replicas.

This used to live in SessionWriter's memory, so a redeploy lost the segment: an
unclosed segment could never be closed, leaving a hole in what got remembered, and
with several replicas one session forked in two. Now the pod is stateless:

- RedisSegStore is the production implementation. A segment is one STRING key
  holding the whole segment as JSON, and save uses `SET ... EX` so a single
  command writes the value and the TTL atomically. No pipelines and no Lua: we
  measured the Redis proxy misframing pipelined responses, so one command could
  read another's reply, whereas the single-command path verified reliable
  throughout. The caller (feed) holds the session write lock, so read-modify-write
  over the whole segment has no concurrent-merge problem.
- MemorySegStore is the single-process implementation, for local scripts, unit
  tests, and single-replica deployments with no Redis configured. It is capped so
  it cannot leak.
"""

from __future__ import annotations

import json
from typing import Protocol

from ..models import EvidenceRecord
from .redis_client import key as _key

# Sliding TTL on segment state, renewed on every save. If a session goes a whole
# day without another utterance the segment state lapses, and rebuilding covers it.
SEG_TTL_S = 24 * 3600


class SegStore(Protocol):
    def load(self, user_id: str, session_id: str) -> list[EvidenceRecord]: ...
    def save(self, user_id: str, session_id: str, records: list[EvidenceRecord]) -> None: ...
    def clear(self, user_id: str, session_id: str) -> None: ...


class RedisSegStore:
    """Segment state in Redis: key = {env}:personos:seg:{user}:{session}, a STRING holding the whole segment."""

    def __init__(self, client, ttl_s: int = SEG_TTL_S):
        self._c = client
        self._ttl = ttl_s

    def _k(self, user_id: str, session_id: str) -> str:
        return _key("seg", user_id, session_id)

    def load(self, user_id: str, session_id: str) -> list[EvidenceRecord]:
        raw = self._c.get(self._k(user_id, session_id))
        if not raw:
            return []
        return [EvidenceRecord.model_validate(x) for x in json.loads(raw)]

    def save(self, user_id: str, session_id: str, records: list[EvidenceRecord]) -> None:
        # One SET carrying EX: value and TTL take effect atomically, leaving no
        # window in which the value exists without a TTL.
        payload = json.dumps([r.model_dump(mode="json") for r in records])
        self._c.set(self._k(user_id, session_id), payload, ex=self._ttl)

    def clear(self, user_id: str, session_id: str) -> None:
        self._c.delete(self._k(user_id, session_id))


class MemorySegStore:
    """In-process segment state.

    A cap keeps it from leaking: over the limit, closed (empty) segments are
    cleared first, and if it is still over, the oldest are dropped. This only ever
    affects a local single replica.
    """

    _CAP = 1024   # maximum number of sessions; each segment holds at most MAX_SEGMENT_TURNS utterances, so total memory is bounded

    def __init__(self):
        self._d: dict[tuple[str, str], list[EvidenceRecord]] = {}

    def load(self, user_id: str, session_id: str) -> list[EvidenceRecord]:
        return list(self._d.get((user_id, session_id), []))

    def save(self, user_id: str, session_id: str, records: list[EvidenceRecord]) -> None:
        k = (user_id, session_id)
        if records:
            self._d[k] = list(records)
        else:
            self._d.pop(k, None)                     # an empty segment holds no slot; equivalent to clear
        if len(self._d) <= self._CAP:
            return
        for stale in [s for s, v in self._d.items() if not v]:
            del self._d[stale]                       # clear closed, empty segments first; normally almost all of them are
            if len(self._d) <= self._CAP:
                return
        while len(self._d) > self._CAP:              # still over: drop the oldest open segments, an extreme case we accept
            self._d.pop(next(iter(self._d)))

    def clear(self, user_id: str, session_id: str) -> None:
        self._d.pop((user_id, session_id), None)
