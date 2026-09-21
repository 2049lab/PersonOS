"""会话写锁:同 (user, session) 的 ingest / session-end 互斥,跨副本生效。

此前是进程内 threading.Lock——多副本下同 session 并发 feed 会交错,甚至两个副本
对同一段各建一个 cell。外置后:
- RedisSessionLock:SET NX PX + 持锁者 token,Lua 比对释放(只删自己的锁)。
  防死锁优先于完美互斥:持锁超 TTL 自动失效,另一副本可进。
- MemorySessionLock:进程内实现(本地/单副本),注册表带上限防泄漏。
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, Protocol

from .redis_client import key as _key

# 持锁 TTL:须盖住「慢而未死」的 feed 最坏耗时——W1 单次 chat 就可能 3 次重试×120s
# 超时≈365s(降速不故障的上游),加 W2 建 cell 余量。锁先过期=另一副本闯入同会话,
# 有重复闭合风险;代价是持锁者崩溃后会话被锁最多 TTL(防死锁优先,如实接受)。
LOCK_TTL_S = 600
# 抢锁等待上界:超时抛 TimeoutError,任务如实报"会话忙"而不是无限排队
LOCK_WAIT_S = 60
_RETRY_S = 0.2

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# 续期:仍持有(token 匹配)才延命,防误续别人的锁。单 EVAL(corvus 支持,与 release 同款)。
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class SessionLock(Protocol):
    def __call__(self, user_id: str, session_id: str): ...   # -> ContextManager[None]
    def try_acquire(self, user_id: str, session_id: str): ...  # -> handle | None(非阻塞)
    def release(self, user_id: str, session_id: str, handle) -> None: ...
    def renew(self, user_id: str, session_id: str, handle) -> bool: ...  # 续期,仍持有才成功


class RedisSessionLock:
    """corvus 分布式锁:key = {env}:personos:lock:{user}:{session}。"""

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
                raise TimeoutError(f"session 写锁等待超时({self._wait}s): {user_id}/{session_id}")
            time.sleep(_RETRY_S)
        try:
            yield
        finally:
            self._c.eval(_RELEASE_LUA, 1, k, token)

    def try_acquire(self, user_id: str, session_id: str) -> str | None:
        """非阻塞抢锁:成功返回 token(release 凭它删自己的锁),失败返回 None(别处正持有)。

        dispatcher 专用——抢不到立刻返回去处理别的会话,绝不自旋占线程。
        """
        k = _key("lock", user_id, session_id)
        token = uuid.uuid4().hex
        return token if self._c.set(k, token, nx=True, px=self._ttl * 1000) else None

    def release(self, user_id: str, session_id: str, handle) -> None:
        """凭 token 释放(Lua 比对,只删自己的);token 为空则不操作。"""
        if not handle:
            return
        self._c.eval(_RELEASE_LUA, 1, _key("lock", user_id, session_id), handle)

    def renew(self, user_id: str, session_id: str, handle) -> bool:
        """续期锁 TTL(仍持有才成功)。长 drain 每处理一条调一次,防锁中途过期被别副本抢入。"""
        if not handle:
            return False
        r = self._c.eval(_RENEW_LUA, 1, _key("lock", user_id, session_id), handle, self._ttl * 1000)
        return bool(r)


class MemorySessionLock:
    """进程内锁注册表(单副本形态)。带上限:超限时清掉当前未持有的锁对象。"""

    _CAP = 4096

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}   # 元组键:段含 "::" 也不碰撞

    @contextmanager
    def __call__(self, user_id: str, session_id: str) -> Iterator[None]:
        k = (user_id, session_id)
        with self._guard:
            lk = self._locks.setdefault(k, threading.Lock())
            if len(self._locks) > self._CAP:     # 超限:清未持有的旧锁(不含刚取的这把)
                for stale in [s for s, l in self._locks.items() if s != k and not l.locked()]:
                    self._locks.pop(stale, None)
        with lk:
            yield

    def _lock_for(self, k):
        with self._guard:
            lk = self._locks.setdefault(k, threading.Lock())
            if len(self._locks) > self._CAP:     # 超限:清未持有的旧锁(try_acquire 路径也不泄漏)
                for stale in [s for s, l in self._locks.items() if s != k and not l.locked()]:
                    self._locks.pop(stale, None)
            return lk

    def try_acquire(self, user_id: str, session_id: str):
        """非阻塞抢锁:成功返回 Lock 句柄(release 用),失败返回 None。"""
        lk = self._lock_for((user_id, session_id))
        return lk if lk.acquire(blocking=False) else None

    def release(self, user_id: str, session_id: str, handle) -> None:
        if handle is not None:
            handle.release()

    def renew(self, user_id: str, session_id: str, handle) -> bool:
        return handle is not None      # 进程内锁不过期,持有即"续期成功"
