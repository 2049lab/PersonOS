"""SessionLock unit tests: mutual-exclusion semantics, plus token-safe release in the Redis implementation
(exercised through FakeRedis).
"""

from __future__ import annotations

import threading
import time

import pytest

from personos.storage.session_lock import MemorySessionLock, RedisSessionLock

from .fakes import FakeRedis


def test_memory_mutual_exclusion():
    """Two holders of the same (user, session) lock run serially: the second cannot enter until the first releases."""
    lk = MemorySessionLock()
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def first():
        with lk("u", "s"):
            order.append("a-in")
            entered.set()
            release.wait(timeout=5)
            order.append("a-out")

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(timeout=5)
    with lk("u", "s"):
        order.append("b-in")                     # getting in at all proves a already released
    t.join(timeout=5)
    assert order == ["a-in", "a-out", "b-in"]


def test_memory_different_sessions_do_not_block():
    lk = MemorySessionLock()
    with lk("u", "s1"):
        with lk("u", "s2"):                      # different sessions do not exclude each other
            pass


def test_memory_lock_registry_capped():
    lk = MemorySessionLock()
    lk._CAP = 4
    for i in range(10):                          # released lock objects should be swept away
        with lk("u", f"s{i}"):
            pass
    assert len(lk._locks) <= 4


def test_redis_acquire_release():
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=0.1)
    with lk("u", "s"):
        key = next(iter(c.data))
        assert c.data[key]                       # the key exists while the lock is held
        # Another party tries to grab the lock while it is held: it fails immediately and then times out waiting.
        with pytest.raises(TimeoutError):
            with lk("u", "s"):
                pass
    assert not c.data                            # the key is deleted on release


def test_redis_release_only_own_token():
    """Once the lock has changed hands -- its TTL expired and someone else took it -- the old holder must not delete
    the new holder's lock on the way out.
    """
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=0.1)
    ctx = lk("u", "s")
    ctx.__enter__()
    c.data[next(iter(c.data))] = "别人的token"    # simulate the TTL lapsing and another party taking the lock
    ctx.__exit__(None, None, None)               # the old holder exits
    assert c.data                                # the new holder's lock is still there


def test_redis_wait_retries_until_free():
    """When a held lock is released, a waiter should acquire it within the retry window."""
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=3)
    c.set("local:personos:lock:u:s", "占坑", nx=True, px=60000)

    def free_soon():
        time.sleep(0.4)
        c.delete("local:personos:lock:u:s")

    threading.Thread(target=free_soon).start()
    t0 = time.monotonic()
    with lk("u", "s"):
        assert time.monotonic() - t0 >= 0.3      # it really waited instead of succeeding immediately


def test_try_acquire_non_blocking():
    """The non-blocking acquire used by the dispatcher: a second attempt while held returns None, and the lock can
    be taken again once released.
    """
    c = FakeRedis()
    lk = RedisSessionLock(c)
    tok = lk.try_acquire("u", "s")
    assert tok
    assert lk.try_acquire("u", "s") is None       # already held, so the non-blocking attempt fails at once
    lk.release("u", "s", tok)
    assert lk.try_acquire("u", "s")               # can be taken again after release


def test_renew_extends_only_when_held_redis():
    """H2: renewal only succeeds while the lock is still held, i.e. the token matches. A wrong token or an already
    released lock fails.
    """
    c = FakeRedis()
    lk = RedisSessionLock(c, ttl_s=100)
    tok = lk.try_acquire("u", "s")
    assert tok and lk.renew("u", "s", tok) is True
    assert lk.renew("u", "s", "wrong-token") is False   # a non-holder cannot renew
    lk.release("u", "s", tok)
    assert lk.renew("u", "s", tok) is False             # already released, so renewal fails


def test_renew_memory_lock():
    lk = MemorySessionLock()
    h = lk.try_acquire("u", "s")
    assert h is not None and lk.renew("u", "s", h) is True
    assert lk.renew("u", "s", None) is False
    lk.release("u", "s", h)
