"""测试替身:不打真实 MAAS,保证单测确定性(dev 原则 §2)。"""

from __future__ import annotations

import numpy as np


class FakeLLM:
    """按队列返回预设的 chat 响应;记录收到的最后一次 user prompt 便于断言。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.last_user_prompt: str | None = None

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        self.last_user_prompt = messages[-1]["content"]
        resp = self._responses.pop(0) if self._responses else '{"ops":[]}'
        # 支持 callable:让后一轮响应能引用上一轮 prompt 里出现的原子 id
        return resp(self.last_user_prompt) if callable(resp) else resp


class FakeEmbedder:
    """返回固定维度的伪向量,种子由文本长度决定(确定性)。"""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([
            np.random.default_rng(len(t)).standard_normal(self.dim).astype(np.float32)
            for t in texts
        ])


class FakeRedis:
    """redis 客户端的最小内存替身:只实现 seg_store/session_lock 用到的命令子集。

    TTL 不真过期(记录下来供断言);eval 只实现「比对 token 删锁」语义。
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
        """redis TTL 语义:无键 -2;有键无 TTL -1(命名避开测试属性 ttl 字典)。"""
        if key not in self.data:
            return -2
        return self.ttl.get(key, -1)

    def eval(self, script, num_keys, key, *args):   # noqa: A002  redis-py 同名参数
        """实现 release(值==token 才删)与 renew(值==token 才续 TTL)两种锁 Lua。"""
        token = args[0] if args else None
        if self.data.get(key) != token:
            return 0
        if "pexpire" in script:                    # renew:续 TTL
            self.ttl[key] = int(args[1]) // 1000 if len(args) > 1 else self.ttl.get(key, -1)
            return 1
        del self.data[key]                          # release:删锁
        return 1

    # —— msg_queue 用到的 List / Set / 计数命令(最小内存语义,贴合真 redis) ——
    def lpush(self, key, *values):
        lst = self.data.get(key)
        if not isinstance(lst, list):
            lst = self.data[key] = []
        for v in values:              # 逐个压左端(与 redis LPUSH 多值语义一致)
            lst.insert(0, v)
        return len(lst)

    def rpoplpush(self, src, dst):
        s = self.data.get(src)
        if not isinstance(s, list) or not s:
            return None
        v = s.pop()                   # 右端弹出(最旧)
        d = self.data.get(dst)
        if not isinstance(d, list):
            d = self.data[dst] = []
        d.insert(0, v)                # 左端压入目标
        return v

    def lrem(self, key, count, value):
        lst = self.data.get(key)
        if not isinstance(lst, list):
            return 0
        removed = 0
        n = count if count > 0 else len(lst)    # count>0:从头移除至多 count 个(本模块只用 1)
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
        e = len(lst) if end == -1 else end + 1  # redis end 含端;-1=到末尾
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

    def expire(self, key, seconds):
        if key in self.data:
            self.ttl[key] = seconds
            return True
        return False
