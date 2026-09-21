"""多租户隔离:两个 user 共库,store 层 user_id 绑定必须保证 A 的数据 B 完全看不见。

这条测试同时是将来 MySQL 迁移的隔离守卫——所有 store 读路径全覆盖。
"""

from personos.models import EvidenceRecord, MemoryAtom
from personos.online.session_context import build_history
from personos.storage.atom_store import AtomStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from personos.storage.session_store import SessionContextStore
from personos.storage.user_store import UserStore
from tests.fakes import FakeEmbedder, FakeLLM


def test_cross_user_isolation(db):
    """A 写入的证据/原子/摘要,B 的绑定 store 一概读不到;同 sha 不跨 user 去重。"""
    ev_a, at_a = EvidenceStore(db, user_id="A"), AtomStore(db, user_id="A")
    ev_b, at_b = EvidenceStore(db, user_id="B"), AtomStore(db, user_id="B")

    emb = FakeEmbedder()
    eid = ev_a.append(EvidenceRecord(holder="user", content_inline="A用户的秘密",
                                     source={"session_id": "s1"}), embedding=emb.embed(["x"])[0])
    a_atom = MemoryAtom(object_type="fact", text="A用户的事实", lifecycle_status="active")
    at_a.upsert(a_atom, embedding=emb.embed(["x"])[0])
    SessionContextStore(db, user_id="A").save("s1", "A的摘要", 2)

    # B 视角:全部为空/查无
    assert ev_b.list() == [] and ev_b.all_with_embeddings() == []
    assert ev_b.get(eid) is None
    assert ev_b.by_session("s1") == [] and ev_b.in_session("s1") == []
    assert ev_b.session_stats() == {}
    assert ev_b.reply_for(eid) is None
    assert at_b.list() == [] and at_b.all_with_embeddings() == []
    assert at_b.get(a_atom.id) is None and at_b.get_embedding(a_atom.id) is None
    assert SessionContextStore(db, user_id="B").get("s1") == ("", 0)

    # 同内容不跨 user 去重:B 写同 sha 证据,应作为 B 的新证据存在
    eid_b = ev_b.append(EvidenceRecord(holder="user", content_inline="A用户的秘密",
                                       source={"session_id": "s1"}))
    assert eid_b != eid and ev_b.get(eid_b) is not None

    # A 仍只看到自己的(B 的同内容证据不进 A 视野)
    assert len(ev_a.list()) == 1 and ev_a.get(eid_b) is None


def test_user_store_register_and_token(db):
    us = UserStore(db)
    base = us.count()  # 共享 SIT 库有真实注册行,断言用增量而非绝对值
    r = us.register("linwan")
    assert r["user_id"] == "linwan" and r["token"]
    assert us.user_id_by_token(r["token"]) == "linwan"
    assert us.user_id_by_token("不存在") is None
    try:
        us.register("linwan")
        assert False, "重复 user_id 应报错"
    except ValueError:
        pass
    auto = us.register()
    assert auto["user_id"].startswith("u_") and us.count() == base + 2


def test_build_history_per_user(db):
    """会话历史构建(user 维度):A 的会话历史 B 读不到,摘要按 (user, session) 隔离。"""
    ev_a = EvidenceStore(db, user_id="A")
    ev_a.append(EvidenceRecord(holder="user", content_inline="A说了一句话", source={"session_id": "s1"}))
    llm = FakeLLM([])   # 不触发压缩
    h_a = build_history(ev_a, "s1", llm=llm)
    assert any("A说了" in t for _, t in h_a)
    h_b = build_history(EvidenceStore(db, user_id="B"), "s1", llm=llm)
    assert h_b == []
