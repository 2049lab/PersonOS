"""按会话有序的持久消息队列(跨副本、可靠、去重)——ingest 有序消费的地基。

背景:眼镜场景短对话+可打断,同一会话消息高频并发到达。旧路径「submit_task+自旋锁」
只互斥不排序(同会话消息可乱序消费、抢不到锁的任务自旋占满线程池)。这里把顺序与可靠性
下沉到一条持久队列:

- **一会话一队列**:消息 LPUSH 到 mq:{user}:{session}(左进);RPOPLPUSH 从右取(FIFO,旧的先)。
- **可靠出队**:取出即挪到 processing 列(mq:{u}:{s}:proc),**消费成功才 LREM**(ack)。
  worker 崩了消息还在 proc,下轮 recover() 回放——可延后消费,但不丢。
- **看板**:有待处理消息的会话登记进 SET mq:active,供 dispatcher 发现该处理谁;
  队列排空才移除(移除后复查一眼,堵"擦号瞬间来新消息"的孤儿竞态)。
- **不重复**:每条消息带 msg_id(uuid)+ seq(会话内 INCR 单调号);消费侧存 cursor=已应用
  的最大 seq,取到 seq<=cursor 判为重投,直接 ack 跳过(崩溃恢复重取同一条也不会记两遍)。

约束:共享 corvus 集群不留常驻键——所有键带 TTL(24h,与 seg_store 对齐;会话滑动续期,
超一天无活动整条作废,与"尾段到期即弃"一致)。单命令路径(不用 pipeline/Lua,corvus 对
pipeline 响应会错位串包,见 seg_store 模块注释)。
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .redis_client import _esc, key as _key

# 入队锁释放(单 EVAL,corvus 支持;与 session_lock 同款 token 比对):只删自己的锁
_ENQ_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _env() -> str:
    return os.environ.get("XHS_ENV") or "local"


def _sess_tag(user_id: str, session_id: str) -> str:
    """会话哈希标签:corvus(Redis Cluster)只按 {} 内内容分 slot——同会话的 main/proc/seq/cursor
    落同一 slot,RPOPLPUSH 等多键命令才合法(否则 ClusterCrossSlotError)。内层转义防碰撞。"""
    return "{" + _esc(user_id) + ":" + _esc(session_id) + "}"

# 队列键滑动 TTL:与 seg_store 同口径。会话有活动就续期;超一天静默 → 整条作废
QUEUE_TTL_S = 24 * 3600


class EnqueueBusy(RuntimeError):
    """入队锁在期限内抢不到(同会话入队严重争用):调用方应稍后重试。
    丢消息比让调用方重试糟得多——故超时即拒,绝不"硬上"与活跃持锁者并发写坏顺序。"""


@dataclass
class Envelope:
    """队列里的一条消息信封:去重/有序元信息 + 业务载荷 + 原始串(ack 时按原串 LREM)。"""
    msg_id: str
    seq: int
    kind: str                      # "ingest" | "session_end"(收尾任务也排队,按序处理)
    payload: dict = field(default_factory=dict)
    raw: str = ""                  # 入列时的精确 JSON 串;ack 靠它精确移除(不同信封 raw 不同)

    @classmethod
    def from_raw(cls, raw: str) -> "Envelope":
        d = json.loads(raw)
        return cls(msg_id=d["msg_id"], seq=int(d["seq"]), kind=d.get("kind", "ingest"),
                   payload=d.get("payload") or {}, raw=raw)

    @staticmethod
    def make(msg_id: str, seq: int, kind: str, payload: dict) -> str:
        """构造入列 JSON 串。字段顺序固定,保证同一条消息的 raw 稳定可用于 LREM。"""
        return json.dumps({"msg_id": msg_id, "seq": seq, "kind": kind, "payload": payload},
                          ensure_ascii=False, sort_keys=True)


class MsgQueue(Protocol):
    def enqueue(self, user_id: str, session_id: str, payload: dict,
                *, kind: str = "ingest") -> tuple[str, int]: ...
    def active_sessions(self, limit: int = 256) -> list[tuple[str, str]]: ...
    def recover(self, user_id: str, session_id: str) -> list[Envelope]: ...
    def reserve(self, user_id: str, session_id: str) -> Optional[Envelope]: ...
    def ack(self, user_id: str, session_id: str, env: Envelope) -> None: ...
    def cursor_get(self, user_id: str, session_id: str) -> int: ...
    def cursor_set(self, user_id: str, session_id: str, seq: int) -> None: ...
    def mark_failed(self, user_id: str, session_id: str, msg_id: str) -> int: ...
    def clear_failed(self, user_id: str, session_id: str, msg_id: str) -> None: ...
    def depth(self, user_id: str, session_id: str) -> int: ...
    def head_kind(self, user_id: str, session_id: str) -> str: ...
    def is_empty(self, user_id: str, session_id: str) -> bool: ...
    def deactivate_if_empty(self, user_id: str, session_id: str) -> bool: ...


def _member(user_id: str, session_id: str) -> str:
    """看板 SET 成员编码:JSON 数组,无歧义地承载 (user, session)(避免分隔符碰撞)。"""
    return json.dumps([user_id, session_id], ensure_ascii=False)


def _decode_member(m) -> tuple[str, str]:
    if isinstance(m, bytes):
        m = m.decode("utf-8")
    u, s = json.loads(m)
    return u, s


def _s(v) -> Optional[str]:
    """Redis 返回值归一为 str(客户端可能给 bytes)。"""
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, bytes) else v


class RedisMsgQueue:
    """corvus 上的会话队列。键族(均带 TTL):
      mq:{u}:{s}          主队列(List,LPUSH 左进 / RPOPLPUSH 右取 = FIFO)
      mq:{u}:{s}:proc     在途列(List,可靠出队的中转;ack 才移除)
      mqseq:{u}:{s}       会话内单调序号(INCR)
      mqcur:{u}:{s}       消费游标(已应用的最大 seq;去重快跳)
      mq:active           看板(SET,成员=编码后的 (u,s);排空即移除)
    """

    def __init__(self, client, ttl_s: int = QUEUE_TTL_S):
        self._c = client
        self._ttl = ttl_s

    # 同会话诸键用同一 {tag} 落同一 slot(RPOPLPUSH 合法);active 是单键,无需 tag
    def _mk(self, u, s): return f"{_env()}:personos:mq:{_sess_tag(u, s)}"
    def _pk(self, u, s): return f"{_env()}:personos:mq:{_sess_tag(u, s)}:proc"
    def _sk(self, u, s): return f"{_env()}:personos:mqseq:{_sess_tag(u, s)}"
    def _ck(self, u, s): return f"{_env()}:personos:mqcur:{_sess_tag(u, s)}"
    def _fk(self, u, s, msg_id): return f"{_env()}:personos:mqfail:{_sess_tag(u, s)}:{_esc(msg_id)}"
    def _elk(self, u, s): return f"{_env()}:personos:enqlock:{_sess_tag(u, s)}"
    def _ak(self): return _key("mq", "active")

    def enqueue(self, user_id, session_id, payload, *, kind="ingest"):
        # 入队锁(独立于消费锁,毫秒级临界区):把"取号 INCR + 入列 LPUSH"原子化。
        # 否则同会话并发入队时,INCR 定序与 LPUSH 物理落地可能相反 → 消费按物理序取到高 seq、
        # 推进游标后把物理迟到的低 seq 消息误判重投丢弃(有序+不丢双破)。用独立键,绝不与
        # 消费的长 drain 锁争用,故不拖慢 ingest 202。
        # PX(2s)刻意 < spin 兜底(3s):持锁者崩溃时锁 2s 自动失效,等待者能在放弃前抢到,
        # 从而崩溃不触发"硬上"乱序;临界区仅几个 redis 往返(~毫秒),2s 余量足够。
        elk = self._elk(user_id, session_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + 3.0
        while not self._c.set(elk, token, nx=True, px=2000):
            if time.monotonic() >= deadline:
                # 超时即拒(不硬上):deadline(3s) > PX(2s) 保证崩溃持锁者的锁必已过期,
                # 走到这里 = 同会话入队严重争用(活跃持锁者)→ 抛给上层回 503 让调用方重试。
                # 绝不与活跃持锁者并发 INCR+LPUSH(那会写坏物理序 → 低 seq 被判重投丢弃)。
                raise EnqueueBusy(f"入队锁争用超时: {user_id}/{session_id}")
            time.sleep(0.003)
        try:
            seq = int(self._c.incr(self._sk(user_id, session_id)))
            if seq == 1:
                # 新纪元(seq 键此前不存在/闲置过期):清残留旧游标,防 seq 重置到 1 撞陈旧高游标丢消息。
                self._c.delete(self._ck(user_id, session_id))
            msg_id = uuid.uuid4().hex
            raw = Envelope.make(msg_id, seq, kind, payload)
            mk = self._mk(user_id, session_id)
            self._c.lpush(mk, raw)                                 # 左进
            self._c.sadd(self._ak(), _member(user_id, session_id))  # 上看板
            # 续期:主队列/序号/看板(游标在 cursor_set 续期;proc 在 reserve/recover 续期)
            self._c.expire(mk, self._ttl)
            self._c.expire(self._sk(user_id, session_id), self._ttl)
            self._c.expire(self._ak(), self._ttl)
        finally:
            self._c.eval(_ENQ_RELEASE_LUA, 1, elk, token)
        return msg_id, seq

    def active_sessions(self, limit=256):
        members = self._c.smembers(self._ak()) or set()
        # 排序后再截断:SMEMBERS 返回 set 顺序不定,排序使 dispatcher 的 _rr 轮转有确定意义
        # (稳定起点轮转防饥饿);也让"取前 limit 个"在会话数超限时是确定集合而非随机漂移。
        out = sorted(_decode_member(m) for m in members)
        return out[:limit]

    def recover(self, user_id, session_id):
        """回放在途列(上次崩溃遗留、reserve 未 ack 的):按 seq 升序,先处理旧的。"""
        pk = self._pk(user_id, session_id)
        raws = self._c.lrange(pk, 0, -1) or []
        if raws:
            self._c.expire(pk, self._ttl)     # 续期:只被 recover 反复回放的在途消息不至于 24h 后静默丢
        envs = [Envelope.from_raw(_s(r)) for r in raws]
        envs.sort(key=lambda e: e.seq)
        return envs

    def reserve(self, user_id, session_id):
        """可靠取一条:RPOPLPUSH 主队列右端(最旧)→ 在途列。无则 None。"""
        raw = _s(self._c.rpoplpush(self._mk(user_id, session_id), self._pk(user_id, session_id)))
        if raw is None:
            return None
        self._c.expire(self._pk(user_id, session_id), self._ttl)  # 在途列续期
        return Envelope.from_raw(raw)

    def head_kind(self, user_id, session_id):
        """瞥一眼队头(右端=最旧)消息的 kind,**不出队**。供调度按类型分池(视频/文本隔离)。
        队空或解析失败 → 空串(调用方按默认池处理)。"""
        try:
            raw = self._c.lrange(self._mk(user_id, session_id), -1, -1)
            if not raw:
                return ""
            return Envelope.from_raw(_s(raw[0])).kind or ""
        except Exception:  # noqa: BLE001  瞥一眼失败不该影响调度
            return ""

    def ack(self, user_id, session_id, env):
        """确认消费:从在途列精确移除该信封(按原始 raw,移一处)。"""
        self._c.lrem(self._pk(user_id, session_id), 1, env.raw)

    def cursor_get(self, user_id, session_id):
        v = _s(self._c.get(self._ck(user_id, session_id)))
        return int(v) if v else 0

    def cursor_set(self, user_id, session_id, seq):
        self._c.set(self._ck(user_id, session_id), str(int(seq)), ex=self._ttl)

    def mark_failed(self, user_id, session_id, msg_id):
        """记一条消息的累计失败次数(按 msg_id),返回新值。达上限即判毒消息。"""
        fk = self._fk(user_id, session_id, msg_id)
        n = int(self._c.incr(fk))
        self._c.expire(fk, self._ttl)
        return n

    def clear_failed(self, user_id, session_id, msg_id):
        """清失败计数(毒消息判定后调,不留垃圾键;成功路径无键则无操作)。"""
        self._c.delete(self._fk(user_id, session_id, msg_id))

    def depth(self, user_id, session_id):
        return int(self._c.llen(self._mk(user_id, session_id)) or 0)

    def is_empty(self, user_id, session_id):
        return (int(self._c.llen(self._mk(user_id, session_id)) or 0) == 0
                and int(self._c.llen(self._pk(user_id, session_id)) or 0) == 0)

    def deactivate_if_empty(self, user_id, session_id):
        """队列排空 → 移出看板。移除后复查主队列:若刚好来了新消息则补回(堵孤儿竞态)。"""
        if not self.is_empty(user_id, session_id):
            return False
        self._c.srem(self._ak(), _member(user_id, session_id))
        # 复查用 is_empty(含 proc):擦号瞬间若有新消息入 main 或别副本 reserve 到 proc,补回看板,
        # 否则该会话再不会被派、proc 里未 ack 的消息卡到 TTL 丢失(孤儿会话)
        if not self.is_empty(user_id, session_id):
            self._c.sadd(self._ak(), _member(user_id, session_id))
            self._c.expire(self._ak(), self._ttl)
            return False
        return True


class MemoryMsgQueue:
    """进程内实现(本地/单副本/单测)。线程安全:dispatcher 多线程会并发访问同一实例。

    语义与 RedisMsgQueue 逐一对齐;不做 TTL 过期(单进程生命周期内不需要)。
    """

    _CAP = 4096   # 会话元数据上限(单副本长跑防泄漏;超限 GC 已排空的会话)

    def __init__(self):
        self._guard = threading.Lock()
        self._main: dict[tuple[str, str], list[str]] = {}   # 左进右取:index 0=最新,-1=最旧
        self._proc: dict[tuple[str, str], list[str]] = {}
        self._seq: dict[tuple[str, str], int] = {}
        self._cur: dict[tuple[str, str], int] = {}
        self._fail: dict[tuple[str, str, str], int] = {}   # (u,s,msg_id) -> 累计失败次数
        self._active: set[tuple[str, str]] = set()

    def _gc_locked(self):
        """已持 _guard:会话元数据超限时,清掉已排空(main+proc 皆空)会话的残留元数据。"""
        if len(self._seq) <= self._CAP:
            return
        for k in [k for k in self._seq if not self._main.get(k) and not self._proc.get(k)]:
            self._seq.pop(k, None)
            self._cur.pop(k, None)
            for fk in [fk for fk in self._fail if fk[:2] == k]:
                self._fail.pop(fk, None)

    def enqueue(self, user_id, session_id, payload, *, kind="ingest"):
        k = (user_id, session_id)
        with self._guard:
            self._gc_locked()
            self._seq[k] = self._seq.get(k, 0) + 1
            seq = self._seq[k]
            if seq == 1:
                self._cur.pop(k, None)      # 新纪元:清陈旧游标(与 Redis 变体同语义)
            msg_id = uuid.uuid4().hex
            raw = Envelope.make(msg_id, seq, kind, payload)
            self._main.setdefault(k, []).insert(0, raw)         # 左进
            self._active.add(k)
            return msg_id, seq

    def active_sessions(self, limit=256):
        with self._guard:
            return sorted(self._active)[:limit]   # 排序:与 Redis 变体同语义,轮转起点稳定

    def recover(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            envs = [Envelope.from_raw(r) for r in self._proc.get(k, [])]
        envs.sort(key=lambda e: e.seq)
        return envs

    def head_kind(self, user_id, session_id):
        """队头(右端=最旧)的 kind,不出队。语义与 Redis 实现一致。"""
        with self._guard:
            main = self._main.get((user_id, session_id)) or []
            if not main:
                return ""
            try:
                return Envelope.from_raw(main[-1]).kind or ""
            except Exception:  # noqa: BLE001
                return ""

    def reserve(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            main = self._main.get(k) or []
            if not main:
                return None
            raw = main.pop()                                    # 右取(最旧)
            self._proc.setdefault(k, []).insert(0, raw)
            return Envelope.from_raw(raw)

    def ack(self, user_id, session_id, env):
        k = (user_id, session_id)
        with self._guard:
            proc = self._proc.get(k)
            if proc and env.raw in proc:
                proc.remove(env.raw)

    def cursor_get(self, user_id, session_id):
        with self._guard:
            return self._cur.get((user_id, session_id), 0)

    def cursor_set(self, user_id, session_id, seq):
        with self._guard:
            self._cur[(user_id, session_id)] = int(seq)

    def mark_failed(self, user_id, session_id, msg_id):
        with self._guard:
            k = (user_id, session_id, msg_id)
            self._fail[k] = self._fail.get(k, 0) + 1
            return self._fail[k]

    def clear_failed(self, user_id, session_id, msg_id):
        with self._guard:
            self._fail.pop((user_id, session_id, msg_id), None)

    def depth(self, user_id, session_id):
        with self._guard:
            return len(self._main.get((user_id, session_id)) or [])

    def is_empty(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            return not (self._main.get(k) or self._proc.get(k))

    def deactivate_if_empty(self, user_id, session_id):
        k = (user_id, session_id)
        with self._guard:
            if self._main.get(k) or self._proc.get(k):
                return False
            self._active.discard(k)
            if self._main.get(k):                               # 复查:擦号瞬间来了新消息
                self._active.add(k)
                return False
            return True
