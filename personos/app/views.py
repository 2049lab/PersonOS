"""对外视图渲染(纯函数,不 import runtime/不触 MAAS,便于单测)。

对外契约的"记忆视图":干净 + 可溯源,刻意不含任何内部量(打分/RRF 分/prompt/provenance)。
"""

from __future__ import annotations

from ..models import atom_anchor
from ..online.retrieval import _TYPE_CN
from ..online.trust import evidence_entries


def memory_view(a, evidence_store=None, media_store=None) -> dict:
    """一条记忆的对外视图:干净 + 可溯源。

    始终带 evidence_refs(指针);给了 evidence_store 时【内联 evidence】(原文 + 同轮 Q↔A)——
    这样每条记忆都自带溯源证据,对外结构完全一致。图片证据的 evidence 条目带 media_url(原图签名 URL)。
    """
    t = atom_anchor(a)   # 代表时间:occurrence→recorded(与 trust chain 同口径)
    return {
        "atom_id": a.id,
        "cell_id": a.memcell_id or None,     # 归属 cell(可再查段叙事 topic/episode)
        "text": a.text,
        "type": _TYPE_CN.get(a.object_type, a.object_type),   # 经历/事实/说法
        "holder": a.holder,                  # 谁说的/谁的属性(归属)
        "domains": a.domains,
        "kind": a.kind,                      # K 轴记忆类型 code(与 trust chain 对齐)
        "occurred_at": t.isoformat() if t else None,
        "evidence_refs": [r.evidence_id for r in a.evidence_refs],   # 溯源指针
        "evidence": (evidence_entries(a, evidence_store, media_store)
                     if evidence_store is not None else []),
    }
