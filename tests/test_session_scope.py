"""Entry-point id hygiene plus caller session rewriting (the gate for the P1 security-review fix).

Covers:
- Redis key segment escaping: a (user, session) pair containing ":" no longer produces a
  colliding key (before the fix, ("a:b","c") and ("a","b:c") hashed to the same key, which
  meant cross-user read/write through an unclosed segment);
- seg/lock storage and isolation under ids containing ":" (works normally, and the two never
  cross);
- MemorySessionLock tuple keys: the same collision no longer holds;
- scoped_session: caller rewriting / empty-caller compatibility / rejecting illegal input /
  length alignment;
- valid_user_id: the character set allowed in a self-chosen user_id.
"""

from __future__ import annotations

import os
import threading

import pytest

from personos.session_scope import scoped_session, valid_user_id
from personos.models import EvidenceRecord
from personos.storage.redis_client import key as rkey
from personos.storage.seg_store import RedisSegStore
from personos.storage.session_lock import MemorySessionLock

from .fakes import FakeRedis


def _rec(text: str) -> EvidenceRecord:
    return EvidenceRecord(id=f"ev-{abs(hash(text))}", holder="user", content_inline=text)


# -- Redis key segment escaping --

def test_key_escapes_colon_no_collision():
    """The fix: (u="a:b", s="c") and (u="a", s="b:c") must be two different keys."""
    assert rkey("seg", "a:b", "c") != rkey("seg", "a", "b:c")
    assert rkey("lock", "alice", "s1:s2") != rkey("lock", "alice:s1", "s2")


def test_key_escapes_percent_unambiguously():
    """The escaping is itself unambiguous: a literal "%3A" has its % escaped to %25 first, so it
    cannot be mistaken for a colon that the escaping produced."""
    assert rkey("seg", "a%3A", "c") != rkey("seg", "a:b", "c")    # "a%253Ac" vs "a%3Ab"


def test_key_normal_ids_stay_readable():
    """Ordinary ids are unaffected: keys stay human-readable because only the problem
    characters get escaped."""
    os.environ.pop("PERSONOS_ENV", None)
    assert rkey("seg", "u1", "chat-001") == "local:personos:seg:u1:chat-001"


def test_seg_store_colon_ids_isolated():
    """Segment storage works normally with ids containing colons, and a colliding pair stays
    mutually invisible — this was the direct attack surface before the fix."""
    c = FakeRedis()
    st = RedisSegStore(c, ttl_s=3600)
    st.save("alice", "s1:s2", [_rec("受害者的话")])
    st.save("alice:s1", "s2", [_rec("攻击者的话")])
    assert [r.content_inline for r in st.load("alice", "s1:s2")] == ["受害者的话"]
    assert [r.content_inline for r in st.load("alice:s1", "s2")] == ["攻击者的话"]
    st.clear("alice:s1", "s2")                          # the attacker clears their own key
    assert st.load("alice", "s1:s2") != []              # the victim's segment is untouched
    assert len(c.data) == 1                             # only the victim's single key is left in the store


def test_memory_lock_colon_ids_do_not_collide():
    """In-process lock tuple keys: holding (u="a:b", s="c") must not block (u="a", s="b:c")."""
    lk = MemorySessionLock()
    with lk("a:b", "c"):
        entered = threading.Event()

        def other():
            with lk("a", "b:c"):                        # a different key, so it must not block
                entered.set()

        t = threading.Thread(target=other)
        t.start()
        assert entered.wait(timeout=5)
        t.join(timeout=5)


# -- scoped_session / valid_user_id --


def test_scoped_session_prefixes_caller():
    assert scoped_session("agent-x", "chat-001") == "agent-x:chat-001"


def test_scoped_session_empty_caller_passthrough():
    """No caller means the old caller behaviour is preserved exactly (backward compatible: the
    session_id is unchanged)."""
    assert scoped_session("", "chat-001") == "chat-001"
    assert scoped_session(None, "chat-001") == "chat-001"


def test_scoped_session_rejects_bad_ids():
    with pytest.raises(ValueError):
        scoped_session("agent:x", "chat-001")           # caller contains a colon
    with pytest.raises(ValueError):
        scoped_session("agent x", "chat-001")           # caller contains a space
    with pytest.raises(ValueError):
        scoped_session("a" * 33, "chat-001")            # caller too long
    with pytest.raises(ValueError):
        scoped_session("agent", "s:s")                  # session contains a colon (the key separator)
    with pytest.raises(ValueError):
        scoped_session("agent", "s/x")                  # session contains a slash
    with pytest.raises(ValueError):
        scoped_session("agent", "")                     # session is empty
    with pytest.raises(ValueError):
        scoped_session("agent", "x" * 96)               # session too long


def test_scoped_session_max_length_fits_column():
    """caller (32) + ":" (1) + session (95) = 128, which lines up with the column width of
    memcells / session_context."""
    sid = scoped_session("a" * 32, "b" * 95)
    assert len(sid) == 128


def test_valid_user_id_rules():
    assert valid_user_id(None) is None                  # generated automatically
    assert valid_user_id("  ") is None                  # blank -> generated automatically
    assert valid_user_id("linwan") == "linwan"
    assert valid_user_id("u_01M22QWG") == "u_01M22QWG"
    with pytest.raises(ValueError):
        valid_user_id("alice:s1")                       # colon (Redis key segment injection)
    with pytest.raises(ValueError):
        valid_user_id("a/b")
    with pytest.raises(ValueError):
        valid_user_id("x" * 129)                        # exceeds VARCHAR(128)
