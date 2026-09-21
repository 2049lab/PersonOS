"""信任链装配:采纳的原子 → 归属/认识状态 → 下钻到原始证据内容。

随答案一起返回给使用端,让对方能判断"这个答案可信不可信"(可解释/可信)。
atom→evidence 的指针本就存在(evidence_refs),这里只是把它组装成可读结构主动吐出来。
"""

from __future__ import annotations

from ..storage.atom_store import AtomStore
from ..storage.evidence_store import EvidenceStore


def _media_url(ev, media_store) -> str | None:
    """图片证据 → 原图签名 URL(供前端展示);非图/无留底/无签名器 → None。"""
    if media_store is None or ev.modality not in ("image", "mixed") or not ev.content_ref:
        return None
    try:
        return media_store.sign_url(ev.content_ref)
    except Exception:   # noqa: BLE001  签名失败不影响溯源主体
        return None


def evidence_entries(atom, evidence_store: EvidenceStore, media_store=None) -> list[dict]:
    """把一条原子的 evidence_refs 装配成证据列表:每条含原文 + 同轮助手回复(Q↔A 配对)。

    对外记忆视图(recall 的 memories)与信任链(trust chain / trace)共用,保证证据子结构一致。
    图片证据额外带 modality 与 media_url(原图签名 URL);media_store 未给则只标 modality。
    """
    evs = []
    for ref in atom.evidence_refs:
        ev = evidence_store.get(ref.evidence_id)
        if ev is None:
            continue
        entry = {
            "id": ev.id,
            "holder": ev.holder,
            "content": ev.content_inline,
            "captured_at": ev.captured_at.isoformat() if ev.captured_at else None,
        }
        if ev.modality != "text":
            entry["modality"] = ev.modality
            url = _media_url(ev, media_store)
            if url:
                entry["media_url"] = url
        reply = evidence_store.reply_for(ev.id)   # 同轮助手回复(若有)→ 完整 Q↔A
        if reply is not None:
            entry["reply"] = {
                "id": reply.id,
                "content": reply.content_inline,
                "captured_at": reply.captured_at.isoformat() if reply.captured_at else None,
            }
        evs.append(entry)
    return evs


def _ev_dict(ev, media_store=None) -> dict:
    d = {"id": ev.id, "holder": ev.holder, "content": ev.content_inline,
         "captured_at": ev.captured_at.isoformat() if ev.captured_at else None}
    if ev.modality != "text":
        d["modality"] = ev.modality
        url = _media_url(ev, media_store)
        if url:
            d["media_url"] = url
    return d


def trace_evidence(evidence_id: str, evidence_store: EvidenceStore, atom_store: AtomStore,
                   media_store=None) -> dict | None:
    """按 evidence_id 反向溯源:还原【整轮对话对】(固定 user→assistant 顺序)+ 【被哪些记忆引用】。

    不管查的是用户那半还是助手那半,都补齐整对:
    - 查到 user 句 → 配上同轮 assistant 回复(reply_to 指向它的那条);
    - 查到 assistant 句 → 顺 reply_to 找回它回应的 user 句。
    cited_by 基于【用户那半】(记忆引用的是用户陈述)。与 build_trust_chain(atom→evidence 正向)对偶。查无 → None。
    """
    ev = evidence_store.get(evidence_id)
    if ev is None:
        return None
    # 定位这一轮的 user / assistant 两半
    if ev.holder == "assistant":
        user_ev = evidence_store.get((ev.source or {}).get("reply_to", "") or "")
        asst_ev = ev
    else:
        user_ev = ev
        asst_ev = evidence_store.reply_for(ev.id)
    pair = [_ev_dict(e, media_store) for e in (user_ev, asst_ev) if e is not None]   # user → assistant
    anchor_id = user_ev.id if user_ev is not None else evidence_id      # 记忆引用的是用户那半
    cited_by = [a.id for a in atom_store.list(limit=10000)
                if any(r.evidence_id == anchor_id for r in a.evidence_refs)]
    return {"node": "evidence", "evidence_id": ev.id, "pair": pair, "cited_by": cited_by}


def build_trust_chain(
    atom_ids: list[str],
    atom_store: AtomStore,
    evidence_store: EvidenceStore,
    media_store=None,
) -> list[dict]:
    """把采纳原子 id 列表组装成信任链:每条含归属/认识状态 + 下钻到的证据原文。查不到标 missing(兜底不崩)。"""
    chain: list[dict] = []
    seen: set[str] = set()
    for aid in atom_ids:
        if not aid or aid in seen:
            continue
        seen.add(aid)
        a = atom_store.get(aid)
        if a is None:
            chain.append({"atom_id": aid, "missing": True})
            continue
        evs = evidence_entries(a, evidence_store, media_store)   # 证据+Q↔A(与 recall memories 共用装配)
        chain.append({
            "atom_id": a.id,
            "text": a.text,
            "cell_id": a.memcell_id or None,  # 归属 cell(可再查段叙事)
            "object_type": a.object_type,
            "holder": a.holder,               # 谁说的/谁的属性(归属)
            "domains": a.domains,
            "kind": a.kind,
            "evidence": evs,
            "evidence_count": len(evs),       # 独立证据条数——可信度的直观信号
        })
    return chain
