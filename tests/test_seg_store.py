"""SegStore unit tests: the in-process implementation and the Redis one, the latter against a
FakeRedis double rather than a real cluster.

Key shape of the Redis implementation: a single STRING holds the whole segment as JSON, so one
SET ... EX command sets the value and the TTL atomically. Pipelining is deliberately not used,
because the Redis proxy in front of the cluster can misalign pipelined responses; see the
module comments in seg_store.
"""

from __future__ import annotations

from personos.models import EvidenceRecord
from personos.storage.seg_store import MemorySegStore, RedisSegStore

from .fakes import FakeRedis


def _rec(text: str) -> EvidenceRecord:
    return EvidenceRecord(id=f"ev-{text}", holder="user", content_inline=text)


def test_memory_roundtrip_and_clear():
    st = MemorySegStore()
    st.save("u1", "s1", [_rec("你好"), _rec("第二句")])
    st.save("u1", "s2", [_rec("另一会话")])
    assert [r.content_inline for r in st.load("u1", "s1")] == ["你好", "第二句"]
    assert [r.content_inline for r in st.load("u1", "s2")] == ["另一会话"]
    st.clear("u1", "s1")
    assert st.load("u1", "s1") == []
    assert st.load("u1", "s2") != []            # other sessions are unaffected
    st.clear("u1", "s1")                        # idempotent
    assert st.load("u1", "s1") == []


def test_memory_save_empty_releases_slot():
    """Saving an empty list means there is no segment, and it takes no slot in the registry;
    it is equivalent to clear."""
    st = MemorySegStore()
    st.save("u1", "s1", [_rec("x")])
    st.save("u1", "s1", [])
    assert st.load("u1", "s1") == []
    assert ("u1", "s1") not in st._d


def test_memory_user_isolation():
    st = MemorySegStore()
    st.save("u1", "s", [_rec("u1 的话")])
    st.save("u2", "s", [_rec("u2 的话")])
    assert [r.content_inline for r in st.load("u1", "s")] == ["u1 的话"]
    st.clear("u2", "s")
    assert st.load("u1", "s") != []


def test_memory_cap_bounds_growth():
    st = MemorySegStore()
    st._CAP = 8                                  # lower the cap on this instance to exercise eviction
    for i in range(20):
        st.save("u", f"s{i}", [_rec(f"第{i}句")])
    assert len(st._d) <= 8
    assert len(st.load("u", "s19")) == 1         # the newest session is still there


def test_memory_cap_prefers_dropping_closed_segments():
    st = MemorySegStore()
    st._CAP = 4
    for i in range(4):                           # 4 closed (empty) segments
        st.save("u", f"old{i}", [_rec("x")])
        st.clear("u", f"old{i}")
    for i in range(4):                           # 4 open segments
        st.save("u", f"open{i}", [_rec("x")])
    st.save("u", "open3", [_rec("x"), _rec("y")])   # goes over the cap: closed segments must be dropped first
    assert len(st._d) <= 4
    assert len(st.load("u", "open3")) == 2       # the open segment was not dropped by mistake


def test_redis_roundtrip_with_ttl():
    c = FakeRedis()
    st = RedisSegStore(c, ttl_s=3600)
    st.save("u1", "s1", [_rec("你好")])
    st.save("u1", "s1", [_rec("你好"), _rec("第二句")])   # overwriting is how appending works: feed reads, modifies and writes under the lock
    assert [r.content_inline for r in st.load("u1", "s1")] == ["你好", "第二句"]
    key = next(iter(c.data))                     # there is exactly one STRING key
    assert c.ttl[key] == 3600                    # every save renews it, giving a sliding TTL
    st.clear("u1", "s1")
    assert st.load("u1", "s1") == []
    assert key not in c.data


def test_redis_load_missing_key_empty():
    c = FakeRedis()
    st = RedisSegStore(c)
    assert st.load("nobody", "s") == []          # a missing key means an empty segment, not an exception
