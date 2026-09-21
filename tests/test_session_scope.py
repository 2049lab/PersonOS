"""入口 id 卫生 + 调用方 session 改写单测(安全审查 P1 修复的守门)。

覆盖:
- redis key 段转义:含 ":" 的 (user,session) 对不再拼出碰撞键(修复前
  ("a:b","c") 与 ("a","b:c") 是同一个键 = 跨用户串读写未闭合段);
- seg/lock 在含 ":" id 下的存取与隔离(存取正常 + 互不串);
- MemorySessionLock 元组键:同款碰撞不再成立;
- scoped_session:caller 改写/空 caller 兼容/非法输入拒绝/长度对齐;
- valid_user_id:自选 user_id 字符集。
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


# —— redis key 段转义 ——

def test_key_escapes_colon_no_collision():
    """修复点:(u="a:b", s="c") 与 (u="a", s="b:c") 必须是两个不同的键。"""
    assert rkey("seg", "a:b", "c") != rkey("seg", "a", "b:c")
    assert rkey("lock", "alice", "s1:s2") != rkey("lock", "alice:s1", "s2")


def test_key_escapes_percent_unambiguously():
    """转义自身无歧义:字面 "%3A" 先把 % 转义成 %25,不会被误解成转义出来的冒号。"""
    assert rkey("seg", "a%3A", "c") != rkey("seg", "a:b", "c")    # "a%253Ac" vs "a%3Ab"


def test_key_normal_ids_stay_readable():
    """常规 id 不受影响:键保持人类可读的形态(只转义出问题的字符)。"""
    os.environ.pop("PERSONOS_ENV", None)
    assert rkey("seg", "u1", "chat-001") == "local:personos:seg:u1:chat-001"


def test_seg_store_colon_ids_isolated():
    """seg 存取在含冒号 id 下正常,且碰撞对互不可见(修复前的直接攻击面)。"""
    c = FakeRedis()
    st = RedisSegStore(c, ttl_s=3600)
    st.save("alice", "s1:s2", [_rec("受害者的话")])
    st.save("alice:s1", "s2", [_rec("攻击者的话")])
    assert [r.content_inline for r in st.load("alice", "s1:s2")] == ["受害者的话"]
    assert [r.content_inline for r in st.load("alice:s1", "s2")] == ["攻击者的话"]
    st.clear("alice:s1", "s2")                          # 攻击者清自己的键
    assert st.load("alice", "s1:s2") != []              # 受害者段不受影响
    assert len(c.data) == 1                             # 库里只剩受害者的一个键


def test_memory_lock_colon_ids_do_not_collide():
    """进程内锁元组键:(u="a:b",s="c") 持锁不挡 (u="a",s="b:c")。"""
    lk = MemorySessionLock()
    with lk("a:b", "c"):
        entered = threading.Event()

        def other():
            with lk("a", "b:c"):                        # 不同键 → 不应阻塞
                entered.set()

        t = threading.Thread(target=other)
        t.start()
        assert entered.wait(timeout=5)
        t.join(timeout=5)


# —— scoped_session / valid_user_id ——


def test_scoped_session_prefixes_caller():
    assert scoped_session("agent-x", "chat-001") == "agent-x:chat-001"


def test_scoped_session_empty_caller_passthrough():
    """caller 缺省 = 旧调用方行为原样(向前兼容,session_id 不变)。"""
    assert scoped_session("", "chat-001") == "chat-001"
    assert scoped_session(None, "chat-001") == "chat-001"


def test_scoped_session_rejects_bad_ids():
    with pytest.raises(ValueError):
        scoped_session("agent:x", "chat-001")           # caller 含冒号
    with pytest.raises(ValueError):
        scoped_session("agent x", "chat-001")           # caller 含空格
    with pytest.raises(ValueError):
        scoped_session("a" * 33, "chat-001")            # caller 超长
    with pytest.raises(ValueError):
        scoped_session("agent", "s:s")                  # session 含冒号(key 分隔符)
    with pytest.raises(ValueError):
        scoped_session("agent", "s/x")                  # session 含斜杠
    with pytest.raises(ValueError):
        scoped_session("agent", "")                     # session 空
    with pytest.raises(ValueError):
        scoped_session("agent", "x" * 96)               # session 超长


def test_scoped_session_max_length_fits_column():
    """caller(32)+":"(1)+session(95) = 128,对齐 memcells/session_context 列宽。"""
    sid = scoped_session("a" * 32, "b" * 95)
    assert len(sid) == 128


def test_valid_user_id_rules():
    assert valid_user_id(None) is None                  # 自动生成
    assert valid_user_id("  ") is None                  # 空白 → 自动生成
    assert valid_user_id("linwan") == "linwan"
    assert valid_user_id("u_01M22QWG") == "u_01M22QWG"
    with pytest.raises(ValueError):
        valid_user_id("alice:s1")                       # 冒号(Redis 键段注入)
    with pytest.raises(ValueError):
        valid_user_id("a/b")
    with pytest.raises(ValueError):
        valid_user_id("x" * 129)                        # 超 VARCHAR(128)
