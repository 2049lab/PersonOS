"""未闭合段(W1 写入状态)的存取:跨请求、跨副本共享的会话态。

此前常驻 SessionWriter 内存——重部署即丢段(未闭合段永远闭不上,记忆覆盖出现洞),
多副本下同一 session 劈叉。外置后 pod 无状态:
- RedisSegStore:生产实现。段 = 单个 STRING 键存整段 JSON,SAVE 用 `SET ... EX`
  一条命令原子写入值+TTL——不走 pipeline/Lua:实测 corvus 对 pipeline 响应会
  错位串包(跨命令读到别人的回包),单命令路径全部验证可靠。调用方(feed)持
  会话写锁,整段读改写不存在并发合并问题。
- MemorySegStore:单进程实现(本地脚本/单测/未配 Redis 的单副本模式),带上限防泄漏。
"""

from __future__ import annotations

import json
from typing import Protocol

from ..models import EvidenceRecord
from .redis_client import key as _key

# 段状态滑动 TTL:每次 save 续期;一个会话超过一天没有下一句,段状态作废由重建兜底
SEG_TTL_S = 24 * 3600


class SegStore(Protocol):
    def load(self, user_id: str, session_id: str) -> list[EvidenceRecord]: ...
    def save(self, user_id: str, session_id: str, records: list[EvidenceRecord]) -> None: ...
    def clear(self, user_id: str, session_id: str) -> None: ...


class RedisSegStore:
    """corvus 上的段状态:key = {env}:personos:seg:{user}:{session},STRING 存整段。"""

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
        # SET 一条命令带 EX:值与 TTL 原子生效,不留"写了值没 TTL"的故障窗口
        payload = json.dumps([r.model_dump(mode="json") for r in records])
        self._c.set(self._k(user_id, session_id), payload, ex=self._ttl)

    def clear(self, user_id: str, session_id: str) -> None:
        self._c.delete(self._k(user_id, session_id))


class MemorySegStore:
    """进程内段状态。上限防泄漏:超限时先清已闭合(空)段,仍超丢最旧(仅影响本地单副本)。"""

    _CAP = 1024   # 会话数上限;每段 ≤ MAX_SEGMENT_TURNS 句,内存总量有界

    def __init__(self):
        self._d: dict[tuple[str, str], list[EvidenceRecord]] = {}

    def load(self, user_id: str, session_id: str) -> list[EvidenceRecord]:
        return list(self._d.get((user_id, session_id), []))

    def save(self, user_id: str, session_id: str, records: list[EvidenceRecord]) -> None:
        k = (user_id, session_id)
        if records:
            self._d[k] = list(records)
        else:
            self._d.pop(k, None)                     # 空段不占位(与 clear 等价)
        if len(self._d) <= self._CAP:
            return
        for stale in [s for s, v in self._d.items() if not v]:
            del self._d[stale]                       # 先清已闭合的空段(常态:绝大多数是空)
            if len(self._d) <= self._CAP:
                return
        while len(self._d) > self._CAP:              # 仍超:丢最旧的开段(极端场景,如实接受)
            self._d.pop(next(iter(self._d)))

    def clear(self, user_id: str, session_id: str) -> None:
        self._d.pop((user_id, session_id), None)
