"""SegStore 单测:进程内实现 + Redis 实现(FakeRedis 替身,不打真集群)。

Redis 实现的键形态:单个 STRING 存整段 JSON(SET ... EX 一条命令原子值+TTL;
corvus 对 pipeline 响应错位,禁用 pipeline 是设计约束,见 seg_store 模块注释)。
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
    assert st.load("u1", "s2") != []            # 别的会话不受影响
    st.clear("u1", "s1")                        # 幂等
    assert st.load("u1", "s1") == []


def test_memory_save_empty_releases_slot():
    """save 空列表 = 无段状态,不占注册表名额(与 clear 等价)。"""
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
    st._CAP = 8                                  # 实例级压低上限,验证淘汰逻辑
    for i in range(20):
        st.save("u", f"s{i}", [_rec(f"第{i}句")])
    assert len(st._d) <= 8
    assert len(st.load("u", "s19")) == 1         # 最新会话仍在


def test_memory_cap_prefers_dropping_closed_segments():
    st = MemorySegStore()
    st._CAP = 4
    for i in range(4):                           # 4 个已闭合(空)段
        st.save("u", f"old{i}", [_rec("x")])
        st.clear("u", f"old{i}")
    for i in range(4):                           # 4 个开段
        st.save("u", f"open{i}", [_rec("x")])
    st.save("u", "open3", [_rec("x"), _rec("y")])   # 触发超限:应先清空段
    assert len(st._d) <= 4
    assert len(st.load("u", "open3")) == 2       # 开段未被误清


def test_redis_roundtrip_with_ttl():
    c = FakeRedis()
    st = RedisSegStore(c, ttl_s=3600)
    st.save("u1", "s1", [_rec("你好")])
    st.save("u1", "s1", [_rec("你好"), _rec("第二句")])   # 覆写 = 追加语义(feed 持锁读改写)
    assert [r.content_inline for r in st.load("u1", "s1")] == ["你好", "第二句"]
    key = next(iter(c.data))                     # 只有一个 STRING 键
    assert c.ttl[key] == 3600                    # 每次 save 都续期(滑动 TTL)
    st.clear("u1", "s1")
    assert st.load("u1", "s1") == []
    assert key not in c.data


def test_redis_load_missing_key_empty():
    c = FakeRedis()
    st = RedisSegStore(c)
    assert st.load("nobody", "s") == []          # 无键 = 空段,不抛
