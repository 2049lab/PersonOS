"""SessionLock 单测:互斥语义 + Redis 实现(FakeRedis)的 token 安全释放。"""

from __future__ import annotations

import threading
import time

import pytest

from personos.storage.session_lock import MemorySessionLock, RedisSessionLock

from .fakes import FakeRedis


def test_memory_mutual_exclusion():
    """同 (user,session) 的两个持锁方串行:后者必须等前者释放才能进。"""
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
        order.append("b-in")                     # 能进 = a 已释放
    t.join(timeout=5)
    assert order == ["a-in", "a-out", "b-in"]


def test_memory_different_sessions_do_not_block():
    lk = MemorySessionLock()
    with lk("u", "s1"):
        with lk("u", "s2"):                      # 不同会话不互斥
            pass


def test_memory_lock_registry_capped():
    lk = MemorySessionLock()
    lk._CAP = 4
    for i in range(10):                          # 已释放的锁对象应被清扫
        with lk("u", f"s{i}"):
            pass
    assert len(lk._locks) <= 4


def test_redis_acquire_release():
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=0.1)
    with lk("u", "s"):
        key = next(iter(c.data))
        assert c.data[key]                       # 持锁期间 key 存在
        # 持锁期间他人抢锁:立即失败 → 等待超时
        with pytest.raises(TimeoutError):
            with lk("u", "s"):
                pass
    assert not c.data                            # 释放后删除


def test_redis_release_only_own_token():
    """锁已易主(TTL 过期被他人抢走)时,旧持有者退出不得误删新锁。"""
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=0.1)
    ctx = lk("u", "s")
    ctx.__enter__()
    c.data[next(iter(c.data))] = "别人的token"    # 模拟 TTL 失效后被他人抢走
    ctx.__exit__(None, None, None)               # 旧持有者退出
    assert c.data                                # 新持有者的锁仍在


def test_redis_wait_retries_until_free():
    """锁被占后释放,等待方应在重试窗口内拿到。"""
    c = FakeRedis()
    lk = RedisSessionLock(c, wait_s=3)
    c.set("local:personos:lock:u:s", "占坑", nx=True, px=60000)

    def free_soon():
        time.sleep(0.4)
        c.delete("local:personos:lock:u:s")

    threading.Thread(target=free_soon).start()
    t0 = time.monotonic()
    with lk("u", "s"):
        assert time.monotonic() - t0 >= 0.3      # 确实等了,而不是立即成功


def test_try_acquire_non_blocking():
    """dispatcher 用的非阻塞抢锁:占用后再抢返回 None,释放后可再抢。"""
    c = FakeRedis()
    lk = RedisSessionLock(c)
    tok = lk.try_acquire("u", "s")
    assert tok
    assert lk.try_acquire("u", "s") is None       # 已占 → 非阻塞立即失败
    lk.release("u", "s", tok)
    assert lk.try_acquire("u", "s")               # 释放后可再抢


def test_renew_extends_only_when_held_redis():
    """H2:续期仅在仍持锁(token 匹配)时成功;错 token / 已释放 → 失败。"""
    c = FakeRedis()
    lk = RedisSessionLock(c, ttl_s=100)
    tok = lk.try_acquire("u", "s")
    assert tok and lk.renew("u", "s", tok) is True
    assert lk.renew("u", "s", "wrong-token") is False   # 非持有者不能续
    lk.release("u", "s", tok)
    assert lk.renew("u", "s", tok) is False             # 已释放,续期失败


def test_renew_memory_lock():
    lk = MemorySessionLock()
    h = lk.try_acquire("u", "s")
    assert h is not None and lk.renew("u", "s", h) is True
    assert lk.renew("u", "s", None) is False
    lk.release("u", "s", h)
