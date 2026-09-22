"""Test doubles: never call the real model gateway, so unit tests stay deterministic."""

from __future__ import annotations

import numpy as np


class FakeLLM:
    """Returns canned chat responses from a queue, and records the last user prompt it saw so tests can assert on it."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.last_user_prompt: str | None = None

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        self.last_user_prompt = messages[-1]["content"]
        resp = self._responses.pop(0) if self._responses else '{"ops":[]}'
        # A response may be a callable, so a later turn can refer to atom ids that showed up in the previous prompt.
        return resp(self.last_user_prompt) if callable(resp) else resp


class FakeEmbedder:
    """Returns fake vectors of a fixed dimension, seeded by text length so the output is deterministic."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([
            np.random.default_rng(len(t)).standard_normal(self.dim).astype(np.float32)
            for t in texts
        ])


class FakeRedis:
    """A minimal in-memory stand-in for the redis client: only the command subset that seg_store and session_lock use.

    TTLs never actually expire -- they are just recorded so tests can assert on them. eval only implements the
    "delete the lock if the token matches" semantics.
    """

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    def set(self, key, value, nx=False, px=None, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        if px is not None:
            self.ttl[key] = px // 1000
        elif ex is not None:
            self.ttl[key] = ex
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.data:
                del self.data[k]
                n += 1
            self.ttl.pop(k, None)
        return n

    def ttl_of(self, key):
        """Redis TTL semantics: -2 when the key is missing, -1 when the key exists without a TTL. Named to avoid
        clashing with this fake's own ttl dict attribute.
        """
        if key not in self.data:
            return -2
        return self.ttl.get(key, -1)

    def eval(self, script, num_keys, key, *args):   # noqa: A002  parameter name matches redis-py
        """Implements both lock Lua scripts: release (delete only when the stored value equals the token) and
        renew (extend the TTL only when the stored value equals the token).
        """
        token = args[0] if args else None
        if self.data.get(key) != token:
            return 0
        if "pexpire" in script:                    # renew: extend the TTL
            self.ttl[key] = int(args[1]) // 1000 if len(args) > 1 else self.ttl.get(key, -1)
            return 1
        del self.data[key]                          # release: drop the lock
        return 1

    # -- List / Set / counter commands used by msg_queue (minimal in-memory semantics that match real redis) --
    def lpush(self, key, *values):
        lst = self.data.get(key)
        if not isinstance(lst, list):
            lst = self.data[key] = []
        for v in values:              # push one at a time onto the left end, matching redis multi-value LPUSH
            lst.insert(0, v)
        return len(lst)

    def rpoplpush(self, src, dst):
        s = self.data.get(src)
        if not isinstance(s, list) or not s:
            return None
        v = s.pop()                   # pop from the right end (the oldest entry)
        d = self.data.get(dst)
        if not isinstance(d, list):
            d = self.data[dst] = []
        d.insert(0, v)                # push onto the left end of the destination
        return v

    def lrem(self, key, count, value):
        lst = self.data.get(key)
        if not isinstance(lst, list):
            return 0
        removed = 0
        n = count if count > 0 else len(lst)    # count>0: remove at most count from the head (this module only uses 1)
        i = 0
        while i < len(lst) and removed < n:
            if lst[i] == value:
                lst.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    def llen(self, key):
        lst = self.data.get(key)
        return len(lst) if isinstance(lst, list) else 0

    def lrange(self, key, start, end):
        lst = self.data.get(key)
        if not isinstance(lst, list):
            return []
        e = len(lst) if end == -1 else end + 1  # redis end is inclusive; -1 means through the last element
        return lst[start:e]

    def sadd(self, key, *members):
        st = self.data.get(key)
        if not isinstance(st, set):
            st = self.data[key] = set()
        before = len(st)
        st.update(members)
        return len(st) - before

    def srem(self, key, *members):
        st = self.data.get(key)
        if not isinstance(st, set):
            return 0
        n = 0
        for m in members:
            if m in st:
                st.discard(m)
                n += 1
        return n

    def smembers(self, key):
        st = self.data.get(key)
        return set(st) if isinstance(st, set) else set()

    def scard(self, key):
        st = self.data.get(key)
        return len(st) if isinstance(st, set) else 0

    def incr(self, key):
        v = int(self.data.get(key, 0)) + 1
        self.data[key] = v
        return v

    def decr(self, key):
        v = int(self.data.get(key, 0)) - 1
        self.data[key] = v
        return v

    def expire(self, key, seconds):
        if key in self.data:
            self.ttl[key] = seconds
            return True
        return False
