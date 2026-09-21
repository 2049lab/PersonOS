"""W2.5 判链测试:预筛 → 分组裁决 → 落库,保守律与失败非阻塞(docs/atom-chain-design.md §4.2)。

不调真实 MAAS:RoutingLLM 按 system prompt 路由,chain assigner 调用喂预设分组。
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from personos.models import ChainInfo, EvidenceRecord, MemoryAtom, now
from personos.online.chain_build import assign_chains
from personos.online.write_path import build_cell
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder

_T0 = datetime(2026, 8, 25, 10, 0)


class RoutingLLM:
    """write_path 三类调用 + chain assigner 四路分发;记录调用序。"""

    def __init__(self, episode=(), atoms=(), assign=()):
        self.q = {"episode": list(episode), "atoms": list(atoms), "assign": list(assign)}

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp
                else "assign" if "chain assigner" in sysp
                else "boundary" if "boundary detector" in sysp else "other")
        assert kind != "other", f"未知 system prompt: {sysp[:40]}"
        if kind == "boundary":
            return '{"should_end": true, "confidence": 0.9, "topic_summary": "t"}'
        assert self.q[kind], f"未预期的 {kind} 调用(队列已空)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


def _items(texts: list[str]):
    """atoms + 各自向量(种子固定,可复现余弦)。"""
    rng = np.random.default_rng(7)
    vecs = [rng.standard_normal(8).astype(np.float32) for _ in texts]
    out = []
    for t, v in zip(texts, vecs):
        a = MemoryAtom(memcell_id="c1", text=t, object_type="fact", occurrence_time=_T0)
        out.append((a, v))
    return out


def _seeded(db, texts: list[str]):
    """W2 语义的 items:先落库(判链的前置是 atoms 已存在),再返回 (atom, vec) 对。"""
    at = AtomStore(db)
    items = _items(texts)
    for a, v in items:
        at.upsert(a, embedding=v)
    return items


def _chain_with_atom(db, title: str, text: str, vec):
    at, cs = AtomStore(db), ChainStore(db)
    a = MemoryAtom(memcell_id="c0", text=text, object_type="fact", occurrence_time=_T0)
    at.upsert(a, embedding=vec)
    return cs.create_chain(ChainInfo(title=title), a, centroid=vec), a


# —— 判链本体 ——

def test_new_chain_grouping(db):
    """零链起步:LLM 把三 atom 归一条新链,成员按格内序、质心=组均值。"""
    cs = ChainStore(db)
    items = _seeded(db, ["Caroline 每周三练热瑜伽", "Caroline 的瑜伽馆在 MBS",
                         "Caroline 说瑜伽课强度很大"])
    llm = RoutingLLM(assign=['{"assignments":[{"chain":"new","title":"Caroline 的瑜伽","atoms":[1,2,3]}]}'])
    res = assign_chains(llm, cs, items, origin_cell_id="cell_x")

    assert res.assigned == 3 and not res.free
    assert len(res.new_chains) == 1
    ch = res.new_chains[0]
    assert ch.title == "Caroline 的瑜伽" and ch.origin_cell_id == "cell_x"
    members = cs.full_chain(ch.id)
    assert [m.text for m in members] == [a.text for a, _ in items]
    _, cen = cs.list_chains()[0]
    assert np.allclose(cen, np.stack([v for _, v in items]).mean(axis=0), atol=1e-6)


def test_append_existing_prefilter_and_centroid(db, rng_vec):
    """归入既有链:预筛候选进 prompt、追加落库、组级增量质心。"""
    at, cs = AtomStore(db), ChainStore(db)
    old_vec = rng_vec(1)
    ch0, _ = _chain_with_atom(db, "Caroline 的瑜伽馆", "Caroline 的瑜伽馆在 MBS", old_vec)

    items = _seeded(db, ["Caroline 的瑜伽馆搬到了 Marina Bay(2026-08-20)",
                         "Caroline 每周三练热瑜伽"])
    # 候选链标签按建链时间序;用 callable 从 prompt 里认出目标链标签
    def pick(prompt):
        label = next(l for l in ("c1", "c2") if f"{l} · Caroline 的瑜伽馆" in prompt)
        return json.dumps({"assignments": [{"chain": label, "atoms": [1]}]})
    llm = RoutingLLM(assign=[pick])
    res = assign_chains(llm, cs, items, origin_cell_id="cell_y")

    assert res.appended == {ch0.id: 1} and len(res.free) == 1
    got = cs.get_chain(ch0.id)
    assert got.n_atoms == 2 and got.tail_atom_id == items[0][0].id
    _, cen = cs.list_chains()[0]
    expect = (old_vec * 1 + items[0][1]) / 2
    assert np.allclose(cen, expect, atol=1e-6)


def test_conservative_hallucinated_label(db):
    """幻觉链标签 → 相关 atom 留游离,不建链不追加(保守律:宁漏勿错)。"""
    cs = ChainStore(db)
    _chain_with_atom(db, "已有链", "已有事实",
                     np.random.default_rng(3).standard_normal(8).astype(np.float32))
    items = _seeded(db, ["一个新事实"])
    llm = RoutingLLM(assign=['{"assignments":[{"chain":"c9","atoms":[1]}]}'])
    res = assign_chains(llm, cs, items)
    assert res.free == [items[0][0].id] and not res.new_chains and not res.appended


def test_parse_failure_all_free(db):
    """LLM 输出非 JSON → 全部留游离,不抛出(失败非阻塞)。"""
    cs = ChainStore(db)
    items = _seeded(db, ["事实A", "事实B"])
    llm = RoutingLLM(assign=["not-json", "still-not-json"])   # num_tries=2 都坏
    res = assign_chains(llm, cs, items, origin_cell_id="c")
    assert sorted(res.free) == sorted(a.id for a, _ in items)
    assert cs.list_chains() == []


def test_unplaced_atoms_default_free(db):
    """LLM 漏掉的 atom 默认游离(每 atom 必须显式归组,漏=新链或游离,不猜)。"""
    cs = ChainStore(db)
    items = _seeded(db, ["事实A", "事实B", "事实C"])
    llm = RoutingLLM(assign=['{"assignments":[{"chain":"new","title":"一组","atoms":[1]}]}'])
    res = assign_chains(llm, cs, items)
    assert len(res.free) == 2 and not any(
        a.id in res.free for a, _ in items[:1])


def test_execution_conflict_isolated(db):
    """单组执行失败(双挂冲突)只丢该组,其余组照常落链。"""
    at, cs = AtomStore(db), ChainStore(db)
    ch_b, b1 = _chain_with_atom(db, "链B", "B 的事实", np.random.default_rng(5).standard_normal(8).astype(np.float32))
    fresh = _seeded(db, ["干净的新事实"])
    items = [(b1, np.random.default_rng(6).standard_normal(8).astype(np.float32))] + fresh
    # 分组:atom1(已在链B)指到链B 追加 → expect_chain 冲突;atom2 开新链应成功
    llm = RoutingLLM(assign=[
        lambda p: json.dumps({"assignments": [
            {"chain": "c1", "atoms": [1]},
            {"chain": "new", "title": "新链", "atoms": [2]}]})])
    res = assign_chains(llm, cs, items)
    assert len(res.free) == 1 and len(res.new_chains) == 1
    assert cs.get_chain(ch_b.id).n_atoms == 1          # 链B 未被污染
    assert cs.full_chain(res.new_chains[0].id)[0].text == "干净的新事实"


# —— build_cell 接线 ——

_EPISODE_OK = ('{"topic": "Caroline 的瑜伽安排", '
               '"episode": "Caroline 每周三练热瑜伽,瑜伽馆在 MBS。", "domains": ["D05"]}')
_ATOMS_OK = ('{"atoms": ['
             '{"text": "Caroline 每周三练热瑜伽", "object_type": "fact", '
             '"holder": "Caroline", "kind": "K05", "domains": ["D05"], '
             '"when": "2026-08-25", "quote": "每周三练热瑜伽"}, '
             '{"text": "Caroline 的瑜伽馆在 MBS", "object_type": "fact", '
             '"holder": "Caroline", "kind": "K01", "domains": ["D05"], '
             '"when": null, "quote": "瑜伽馆在 MBS"}]}')
_ASSIGN_OK = ('{"assignments":[{"chain":"new","title":"Caroline 的瑜伽频率","atoms":[1]},'
              '{"chain":"new","title":"Caroline 的瑜伽地点","atoms":[2]}]}')

_RECORDS = [EvidenceRecord(holder="user", content_inline="我每周三练热瑜伽",
                           source={"session_id": "s"}, captured_at=_T0),
            EvidenceRecord(holder="user", content_inline="瑜伽馆在 MBS",
                           source={"session_id": "s"}, captured_at=_T0)]


def _build(db, llm):
    return build_cell(llm, FakeEmbedder(), EvidenceStore(db), CellStore(db), AtomStore(db),
                      _RECORDS, session_id="s", chain_store=ChainStore(db))


def test_build_cell_empty_assignments_leave_atoms_free(db):
    """LLM 判不出任何归组(assignments 空):atoms 全留游离,写入不阻塞。"""
    llm = RoutingLLM(episode=[_EPISODE_OK], atoms=[_ATOMS_OK],
                     assign=['{"assignments": []}'])
    cb = _build(db, llm)
    assert cb.chain_assign is not None
    assert all(a.chain_id == "" for a in cb.atoms)


def test_build_cell_chains(db):
    """W2 落库后接 W2.5,两条属性各自成链(属性拆分→分链的地基)。"""
    llm = RoutingLLM(episode=[_EPISODE_OK], atoms=[_ATOMS_OK], assign=[_ASSIGN_OK])
    cb = _build(db, llm)
    assert cb.chain_assign is not None
    assert len(cb.chain_assign.new_chains) == 2 and not cb.chain_assign.free
    titles = {c.title for c in cb.chain_assign.new_chains}
    assert titles == {"Caroline 的瑜伽频率", "Caroline 的瑜伽地点"}
    by_text = {a.text: a.chain_id for a, _ in AtomStore(db).all_with_embeddings()}
    assert by_text["Caroline 每周三练热瑜伽"] != by_text["Caroline 的瑜伽馆在 MBS"]
