"""快链单测:R0 五件套解析与降级 / R1 两路 atom 检索+RRF+池选择 / R5 作答引用与降级。

不打真实 MAAS:TableEmbedder 让"文本→向量"完全可控(相似度可手算),FakeLLM 按队列回 JSON。
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from personos.models import MemCell, MemoryAtom
from personos.online.retrieval import (
    AtomHit, CellHit, QueryRewrite, answer_from_cells, cell_block, rewrite_query, search_atoms,
)
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore

from .fakes import FakeLLM

_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


def _v(*x: float) -> np.ndarray:
    return np.array(x, dtype=np.float32)


class TableEmbedder:
    """文本 → 预置向量;表外文本给零向量(绝不调远端)。"""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = {k: _v(*v) for k, v in table.items()}
        self.dim = len(next(iter(self.table.values())))

    def embed(self, texts: list[str]) -> np.ndarray:
        zero = np.zeros(self.dim, dtype=np.float32)
        return np.vstack([self.table.get(t, zero) for t in texts])


class Env:
    """隔离库 + cell/atom store,按需手工建 cell 与带向量的 atom。"""

    def __init__(self, db):
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)

    def add_cell(self, *, topic="t", episode="e", domains=(), atoms=(), topic_vec=None):
        """atoms: [{text, vec, domains?, when?}] → 落库 cell;topic_vec 可选落 topic 向量(topic 路)。"""
        c = MemCell(topic=topic, episode=episode, domains=list(domains),
                    t_start=_T, t_end=_T)
        items = [(MemoryAtom(memcell_id=c.id, text=it["text"],
                             domains=list(it.get("domains", [])),
                             holder=it.get("holder", "user"),
                             object_type=it.get("object_type", "fact"),
                             occurrence_time=it.get("when")), it["vec"])
                 for it in atoms]
        self.cells.upsert(c, topic_embedding=np.asarray(topic_vec, dtype=np.float32)
                          if topic_vec is not None else None)
        self.atoms.upsert_many(items)
        return c

    def hit(self, cell: MemCell, atoms: list[MemoryAtom]) -> CellHit:
        """手工拼 CellHit(R5/R3 材料测试用,不走检索)。"""
        return CellHit(cell=cell, score=0.0, best_sim=0.0,
                       atoms=[AtomHit(atom=a, similarity=0.5) for a in atoms])


def _rw(resolved="q", expansions=(), domains=()):
    return QueryRewrite(original="q", resolved=resolved,
                        expansions=list(expansions), domains=list(domains))


# —— R1 · 两路 atom 检索 + RRF + 池选择 ——

def test_atom_pool_ranked_by_maxsim(db):
    """atom 池按对面(含扩展词)最大 cosine 排;池单元是 atom,不再是 cell。"""
    env = Env(db)
    c1 = env.add_cell(topic="画展", atoms=[
        {"text": "弱相关", "vec": _v(0.6, 0.8, 0, 0)},
        {"text": "强相关", "vec": _v(1, 0, 0, 0)},
    ])
    env.add_cell(topic="跑步机", atoms=[{"text": "中等", "vec": _v(0.8, 0.6, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert [ah.atom.text for ah in pool.atoms] == ["强相关", "中等", "弱相关"]   # 相似度降序
    assert all(ah.atom.memcell_id == c1.id or ah.atom.text == "中等" for ah in pool.atoms)
    assert abs(pool.atoms[0].similarity - 1.0) < 1e-6
    assert pool.atoms[0].rrf > pool.atoms[1].rrf > pool.atoms[2].rrf
    assert pool.beyond == []                                  # 3 atom 全进池,无池外


def test_expansion_face_takes_max(db):
    """联想路查询面各自 embed:atom 分 = 对各面(含扩展词)cosine 的最大值。"""
    env = Env(db)
    env.add_cell(topic="画展", atoms=[{"text": "命中扩展面", "vec": _v(0, 1, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0), "筹备": _v(0, 1, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw(expansions=["筹备"]))
    assert pool.atoms and pool.atoms[0].atom.text == "命中扩展面"
    assert abs(pool.atoms[0].similarity - 1.0) < 1e-6         # 靠扩展面够着


def test_domain_path_and_rrf_dual_hit_wins(db):
    """双路都认的 atom 浮上来:联想路第2+域路第1 的 b,RRF 分超过联想路第1 的单路 c。"""
    env = Env(db)
    env.add_cell(topic="B", domains=["D05"],
                 atoms=[{"text": "b", "vec": _v(0.9, 0.44, 0, 0), "domains": ["D05"]}])
    env.add_cell(topic="C", atoms=[{"text": "c", "vec": _v(1, 0, 0, 0)}])   # 语义最强但无域
    env.add_cell(topic="A", domains=["D05"],
                 atoms=[{"text": "a", "vec": _v(0.5, 0.5, 0.5, 0.5), "domains": ["D05"]}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw(domains=["D05"]))
    assert [ah.atom.text for ah in pool.atoms] == ["b", "a", "c"]   # b(两路) > a(两路,分低) > c(单路第1)
    assert pool.atoms[0].rrf > pool.atoms[2].rrf                   # 两路共识 > 单路榜首
    assert pool.atoms[2].similarity > pool.atoms[0].similarity     # 单路榜首只是 rrf 落后,语义仍最强


def test_no_domains_degrades_to_assoc_order(db):
    """判不出域 → 域路空,RRF 退化为单路:池序 = 联想路相似度序。"""
    env = Env(db)
    env.add_cell(topic="1", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="2", atoms=[{"text": "y", "vec": _v(0.5, 0.5, 0.5, 0.5)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert [ah.atom.text for ah in pool.atoms] == ["x", "y"]


def test_per_cell_cap_blocks_rich_cell(db):
    """同格最多 10 个 atom 进池(防富格灌满挤掉别家);被挤出的不进 beyond(该格已在材料里)。"""
    env = Env(db)
    env.add_cell(topic="富格", atoms=[{"text": f"富{i}", "vec": _v(1 - i * 0.01, 0.1, 0, 0)}
                                      for i in range(12)])
    env.add_cell(topic="弱格", atoms=[{"text": "弱", "vec": _v(0.3, 0.95, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    rich = [ah for ah in pool.atoms if ah.atom.text.startswith("富")]
    assert len(rich) == 10                                   # 富格只进 10 个
    assert [ah.atom.text for ah in pool.atoms[-1:]] == ["弱"]  # 弱格不被挤掉
    assert pool.beyond == []                                 # 池未满,挤出者不算缺料


def test_pool_cut_and_beyond(db):
    """池满(默认 30)后其余名次进 beyond——链面残缺提示的点名原料。"""
    env = Env(db)
    for i in range(33):
        env.add_cell(topic=f"题{i}", atoms=[{"text": f"事实{i}", "vec": _v(1 - i * 0.001, 0.04, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert len(pool.atoms) == 30
    assert [ah.atom.text for ah in pool.beyond] == ["事实30", "事实31", "事实32"]
    assert all(ah.rrf < pool.atoms[-1].rrf for ah in pool.beyond)   # 池外名次低于池尾


def test_orphan_atom_and_empty_pool(db):
    """无 memcell_id 的孤儿 atom 不进池;空库返回空池。"""
    env = Env(db)
    env.atoms.upsert(MemoryAtom(memcell_id="", text="孤儿"), embedding=_v(1, 0, 0, 0))
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})
    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert pool.atoms == [] and pool.beyond == []


# —— R0 · 查询预处理 ——

def test_rewrite_parses_five_outputs(db):
    llm = FakeLLM(['{"resolved":"Caroline 的画展展期定在哪天","subject":"Caroline",'
                   '"expansions":["画展","筹备","展期"],"time_start":"2026-08-11",'
                   '"time_end":"2026-08-17","domains":["D13","D99"]}'])
    rw = rewrite_query(llm, raw_query="她的画展什么时候", now_dt=_T)
    assert rw.resolved == "Caroline 的画展展期定在哪天"
    assert rw.subject == "Caroline"
    assert rw.expansions == ["画展", "筹备", "展期"]
    assert (rw.time_start, rw.time_end) == ("2026-08-11", "2026-08-17")
    assert rw.domains == ["D13"]                    # D99 不在词表 → 丢弃(挡 LLM 自造域)


def test_rewrite_normalizes_nulls(db):
    llm = FakeLLM(['{"resolved":"q","subject":"null","expansions":[],'
                   '"time_start":"null","time_end":null,"domains":[]}'])
    rw = rewrite_query(llm, raw_query="q", now_dt=_T)
    assert rw.subject == "" and rw.time_start == "" and rw.time_end == ""


def test_rewrite_parse_failure_degrades_to_original(db):
    llm = FakeLLM(["模型跑偏,不是 JSON"])
    rw = rewrite_query(llm, raw_query="原始问题", now_dt=_T)
    assert rw.resolved == "原始问题" and rw.domains == [] and rw.expansions == []
    assert "模型跑偏" in rw.raw                     # 溯源:降级也保留模型原文


# —— R5 · 作答 ——

def _cell_with_atoms(env, *, topic, episode, atom_specs):
    c = env.add_cell(topic=topic, episode=episode, atoms=atom_specs)
    return env.hit(c, env.atoms.list_by_cell(c.id))


def test_answer_parses_and_maps_citations(db):
    env = Env(db)
    h1 = _cell_with_atoms(env, topic="画展", episode="叙事一",
                          atom_specs=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0), "when": _T}])
    h2 = _cell_with_atoms(env, topic="跑步", episode="叙事二",
                          atom_specs=[{"text": "跑三公里", "vec": _v(0, 1, 0, 0), "when": _T}])
    llm = FakeLLM(['{"answer":"展期在 2026-09。","cells":["m1","m2","m9"]}'])
    ans = answer_from_cells(llm, query="画展什么时候", subject="user", hits=[h1, h2])
    assert ans.answer == "展期在 2026-09。"
    assert ans.cited_cells == [h1.cell.id, h2.cell.id]        # c9 未知编号 → 丢弃


def test_answer_parse_failure_outputs_raw(db):
    """三次全非 JSON → 原样透出最后一次原文(conv26-v3 实测 MiniMax 偶发非 JSON,R5 曾无重试)。"""
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM(["坏一", "坏二", "坏三"])
    ans = answer_from_cells(llm, query="q", subject="", hits=[h])
    assert ans.answer == "坏三" and ans.cited_cells == []   # 透出的是最后一次原文


def test_answer_infra_error_yields_empty_not_exception_text(db):
    """限流等基建异常:异常文本绝不当答案(H1 实测 'Error code: 429 …' 曾漏成 R5 答案)。"""
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])

    class RateLimitedLLM:
        def chat(self, *a, **k):
            raise RuntimeError("Error code: 429 - {'type': 'rate_limit_error'}")

    ans = answer_from_cells(RateLimitedLLM(), query="q", subject="", hits=[h])
    assert ans.answer == ""                        # 空答 → 上层 _no_answer_note 客观交代


def test_answer_retry_recovers_from_bad_json(db):
    """首输出非 JSON → 重试拿到合法 JSON → 正常作答(bench 实测 7 次失败的对症修复)。"""
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM(["先是一段废话", '{"answer":"重试后的作答。","cells":["m1"]}'])
    ans = answer_from_cells(llm, query="q", subject="", hits=[h])
    assert ans.answer == "重试后的作答。" and ans.cited_cells == [h.cell.id]


def test_answer_empty_hits_skips_llm():
    class CountingLLM:
        calls = 0

        def chat(self, *a, **k):
            self.calls += 1
            return "{}"

    llm = CountingLLM()
    ans = answer_from_cells(llm, query="q", subject="", hits=[])
    assert ans.answer == "" and llm.calls == 0                 # 空材料不调 LLM


def test_answer_prompt_carries_now_anchor(db):
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM(['{"answer":"答","cells":["m1"]}'])
    t0 = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
    answer_from_cells(llm, query="多久了", subject="user", hits=[h], now_dt=t0)
    assert "Current time: 2026-08-28T10:00:00+00:00" in llm.last_user_prompt
    answer_from_cells(llm, query="多久了", subject="user", hits=[h])   # 不传 → 不出该节
    assert "Current time:" not in llm.last_user_prompt


def test_cell_block_renders_material(db):
    """材料格式:「━━ cN/mN ━━」分隔行 + 对话时间/topic 中立头 + episode 主料;atoms 不进材料。
    cell_block 只管渲染,编号由调用方给(快链 mN/深轨 cN 共用此渲染器)。"""
    env = Env(db)
    c = env.add_cell(topic="画展筹备", episode="Caroline 在筹备画展,展期 2026-09。",
                     atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}])
    h = env.hit(c, env.atoms.list_by_cell(c.id))
    block = cell_block(h, "c1")
    lines = block.splitlines()
    assert lines[0] == "━━━ c1 ━━━"                             # 强分隔行(编号在内)
    assert lines[1] == "[dialogue 2026-08-25 | topic: 画展筹备]"
    assert "Caroline 在筹备画展" in block                         # episode 主料
    assert "展期定在 2026-09" not in block                        # atom 文本不进作答材料


# —— R5 · 材料渲染顺序(P1-B 开关)——

def _hit_at(topic: str, episode: str, t):
    """不落库的 CellHit(排序测试只关心 t_start,不需要 store)。"""
    c = MemCell(topic=topic, episode=episode, domains=[], t_start=t, t_end=t)
    return CellHit(cell=c, score=0.0, best_sim=0.0, atoms=[])


def _first_block_episode(llm):
    """从 R5 收到的 user prompt 里抠第一格材料的 episode(分隔行后第二行)。"""
    return llm.last_user_prompt.split("━━━ m1 ━━━\n")[1].splitlines()[1]


def test_r5_order_env_controls_material_order(monkeypatch):
    """PERSONOS_R5_ORDER:relevance=传入(精排)序;time_asc=时间正序;time_desc=倒序;缺省 relevance。"""
    early = _hit_at("早", "一月的事", datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = _hit_at("晚", "六月的事", datetime(2026, 6, 1, tzinfo=timezone.utc))
    llm = FakeLLM(['{"answer":"a","cells":["m1"]}'])

    monkeypatch.setenv("PERSONOS_R5_ORDER", "relevance")
    answer_from_cells(llm, query="q", subject="", hits=[late, early])   # 精排:晚在前
    assert _first_block_episode(llm) == "六月的事"

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_asc")
    answer_from_cells(llm, query="q", subject="", hits=[late, early])
    assert _first_block_episode(llm) == "一月的事"                       # 正序:早的在前

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_desc")
    answer_from_cells(llm, query="q", subject="", hits=[early, late])   # 传入序反转,验证真在排
    assert _first_block_episode(llm) == "六月的事"

    monkeypatch.delenv("PERSONOS_R5_ORDER", raising=False)
    answer_from_cells(llm, query="q", subject="", hits=[late, early])
    assert _first_block_episode(llm) == "六月的事"                       # 缺省 relevance(基线)


def test_r5_order_none_tstart_floor_and_stable_ties(monkeypatch):
    """无 t_start 视为最早垫底;同刻保持传入序(stable)——精排序在同刻内仍是参照。"""
    undated = _hit_at("无期", "无时间的事", None)
    a = _hit_at("同刻A", "A 的事", _T)
    b = _hit_at("同刻B", "B 的事", _T)
    llm = FakeLLM(['{"answer":"a","cells":["m1"]}'])

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_asc")
    answer_from_cells(llm, query="q", subject="", hits=[b, undated, a])
    body = llm.last_user_prompt

    def _ep(handle):
        return body.split(f"━━━ {handle} ━━━\n")[1].splitlines()[1]

    assert _ep("m1") == "无时间的事"      # 无时刻 → 排序下界,最早
    assert _ep("m2") == "B 的事" and _ep("m3") == "A 的事"   # 同刻 stable:保持传入序


def test_answer_prompt_carries_p1_rules():
    """P1 条款护栏:冲突消费的「信最近」与枚举「先数后核」不许后续改版悄悄丢。"""
    from personos.online.retrieval import CONFLICT_RULE, _ANSWER_SYS
    assert "the MOST RECENT statement is the current state" in CONFLICT_RULE
    assert "count the distinct items the materials actually contain" in _ANSWER_SYS
    assert "verify your list has exactly that many" in _ANSWER_SYS
