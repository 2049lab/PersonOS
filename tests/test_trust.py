import pytest
"""信任链单测:采纳原子 → 归属/认识状态 → 下钻证据,装配正确且兜底不崩。"""

from personos.models import EvidenceRecord, EvidenceRef, MemoryAtom
from personos.online.trust import build_trust_chain, trace_evidence
from personos.storage.atom_store import AtomStore
from personos.storage.evidence_store import EvidenceStore


def test_chain_drills_to_evidence(db):
    ev_store = EvidenceStore(db)
    at_store = AtomStore(db)
    eid = ev_store.append(EvidenceRecord(holder="user", content_inline="我上周打篮球扭了右腿"))
    a = MemoryAtom(object_type="event", text="用户上周打篮球扭了右腿", holder="user",
                   domains=["D05"], kind="K01", evidence_refs=[EvidenceRef(evidence_id=eid)])
    at_store.upsert(a)

    chain = build_trust_chain([a.id], at_store, ev_store)
    assert len(chain) == 1
    node = chain[0]
    assert node["text"] == "用户上周打篮球扭了右腿"
    assert node["object_type"] == "event" and node["kind"] == "K01"
    assert node["cell_id"] is None                            # 未挂 cell 的原子不崩
    assert node["evidence_count"] == 1
    assert node["evidence"][0]["content"] == "我上周打篮球扭了右腿"


def test_chain_dedups_ids(db):
    at_store = AtomStore(db)
    a = MemoryAtom(text="无证据原子")
    at_store.upsert(a)
    # 同 id 传两次 → 去重;无证据也不崩
    chain = build_trust_chain([a.id, a.id], at_store, EvidenceStore(db))
    assert len(chain) == 1 and chain[0]["text"] == "无证据原子"
    assert chain[0]["evidence_count"] == 0


def test_evidence_pairs_assistant_reply(db):
    # 信任链的证据配上同轮助手回复(reply_to)→ 外部 agent 溯源时能看到完整 Q↔A,而非单边
    ev_store = EvidenceStore(db)
    at_store = AtomStore(db)
    qid = ev_store.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    ev_store.append(EvidenceRecord(holder="assistant", content_inline="记住了，帮你避开花生",
                                   source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏", evidence_refs=[EvidenceRef(evidence_id=qid)])
    at_store.upsert(a)

    chain = build_trust_chain([a.id], at_store, ev_store)
    e0 = chain[0]["evidence"][0]
    assert e0["content"] == "我对花生过敏"
    assert e0["reply"]["content"] == "记住了，帮你避开花生"     # 配上了当时的回复


def test_missing_atom_marked(db):
    chain = build_trust_chain(["atom_不存在"], AtomStore(db), EvidenceStore(db))
    assert chain[0]["missing"] is True


def test_trace_evidence_reverse(db):
    # 按 evidence_id 反向溯源:还原整轮对(user→assistant 顺序)+ cited_by;查用户半或助手半都补齐整对
    ev_store, at_store = EvidenceStore(db), AtomStore(db)
    qid = ev_store.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    aid = ev_store.append(EvidenceRecord(holder="assistant", content_inline="记住了", source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏", evidence_refs=[EvidenceRef(evidence_id=qid)])
    at_store.upsert(a)

    # 查【用户半】→ 整对,顺序 user→assistant
    node = trace_evidence(qid, ev_store, at_store)
    assert node["node"] == "evidence" and node["evidence_id"] == qid
    assert [p["holder"] for p in node["pair"]] == ["user", "assistant"]        # 顺序固定
    assert [p["content"] for p in node["pair"]] == ["我对花生过敏", "记住了"]
    assert node["cited_by"] == [a.id]                                          # 引用的是用户那半

    # 查【助手半】→ 也补齐同一对,仍是 user→assistant
    node2 = trace_evidence(aid, ev_store, at_store)
    assert [p["holder"] for p in node2["pair"]] == ["user", "assistant"]
    assert node2["cited_by"] == [a.id]                                         # cited_by 基于用户那半
    assert trace_evidence("ev_不存在", ev_store, at_store) is None
