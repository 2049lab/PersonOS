"""对外服务 API 契约:记忆视图【干净 + 可溯源 + 结构一致】——只测纯渲染器(不 import runtime、不触 MAAS)。

端点(ingest/recall/trace)需真实 MAAS + rt 单例,走手动冒烟,不进确定性单测(同 agent 约定)。
"""

from datetime import datetime, timezone

from personos.app.views import memory_view
from personos.models import EvidenceRecord, EvidenceRef, MemoryAtom
from personos.storage.evidence_store import EvidenceStore


def test_memory_view_is_clean_and_traceable():
    a = MemoryAtom(
        object_type="fact", text="我对花生过敏", holder="user",
        memcell_id="cell_1", domains=["D05"], kind="K09",
        occurrence_time=datetime(2026, 7, 13, tzinfo=timezone.utc),
        evidence_refs=[EvidenceRef(evidence_id="ev_1"), EvidenceRef(evidence_id="ev_2")],
    )
    v = memory_view(a)   # 不传 store

    # 可溯源:必带 atom_id + evidence_refs;不传 store 时 evidence 为空列表(结构位仍在)
    assert v["atom_id"] == a.id and v["evidence_refs"] == ["ev_1", "ev_2"] and v["evidence"] == []
    # 语义/归属/时间字段齐全;归属 cell 指针对外暴露(可再查段叙事)
    assert v["type"] == "事实" and v["holder"] == "user" and v["cell_id"] == "cell_1"
    assert v["domains"] == ["D05"] and v["occurred_at"].startswith("2026-07-13")
    assert v["kind"] == "K09"
    # 【不】泄露任何内部量:打分/RRF 分/rerank 分/prompt 不在对外视图里
    for leak in ("breakdown", "score", "similarity", "tier", "salience_score",
                 "provenance", "system", "raw"):
        assert leak not in v, f"对外视图不该含内部字段 {leak}"


def test_memory_view_inlines_evidence_with_qa(db):
    # 给了 evidence_store → 每条记忆内联 evidence(原文 + 同轮 Q↔A);recall 返回的记忆自带溯源
    ev = EvidenceStore(db)
    qid = ev.append(EvidenceRecord(holder="user", content_inline="我对花生过敏"))
    ev.append(EvidenceRecord(holder="assistant", content_inline="记住了，帮你避开花生",
                             source={"reply_to": qid}))
    a = MemoryAtom(object_type="fact", text="用户对花生过敏",
                   evidence_refs=[EvidenceRef(evidence_id=qid)])

    v = memory_view(a, ev)
    assert len(v["evidence"]) == 1
    e0 = v["evidence"][0]
    assert e0["content"] == "我对花生过敏" and e0["reply"]["content"] == "记住了，帮你避开花生"


def test_memory_view_falls_back_to_recorded_at():
    a = MemoryAtom(object_type="claim", text="随口一说")
    v = memory_view(a)
    assert v["occurred_at"] is not None and v["type"] == "说法"
    assert v["cell_id"] is None                            # 未挂 cell → None(不崩)
    assert v["evidence_refs"] == [] and v["evidence"] == []      # 无证据也不崩
