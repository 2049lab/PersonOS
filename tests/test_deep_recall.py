"""深轨单测:渲染器/注册表/六工具(检索走粗排+精排,search_evidence 兜底直搜原话)/MaasChatModel 适配/run_deep agent 循环。

不打真实 MAAS:TableEmbedder 控向量,FakeLLM/桩 LLM 按队列回 JSON 工具调用块驱动 agent 循环。
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from personos.models import ChainInfo, EvidenceRecord, EvidenceRef, MemCell, MemoryAtom
from personos.online.deep_recall import (
    DeepDeps, HandleRegistry, MaasChatModel, build_handoff, cell_full, cell_row,
    evidence_page, run_deep, tool_find_cells, tool_get_cell_evidence, tool_open_cell,
    tool_remember, tool_search_atoms, tool_search_evidence,
)
from personos.online.retrieval import AtomHit, CellHit, QueryRewrite
from personos.online.rerank import NoopReranker
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeLLM

_T = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)


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


class ZeroEmbedder:
    """零向量 embeder(不关心语义的用例用)。"""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class Env:
    """隔离库 + 三 store;add_cell 支持带向量 atoms 与带原话 evidence。"""

    def __init__(self, db):
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)
        self.ev = EvidenceStore(db)

    def add_cell(self, *, topic="t", episode="e", t_start=_T, domains=(),
                 atoms=(), evidence=(), topic_vec=None) -> MemCell:
        """atoms: [{text, vec, domains?, holder?, when?}];evidence: [(holder, 原话)]。"""
        c = MemCell(topic=topic, episode=episode, domains=list(domains),
                    t_start=t_start, t_end=t_start)
        for holder, content in evidence:
            rec = EvidenceRecord(holder=holder, content_inline=content, captured_at=t_start)
            self.ev.append(rec)
            c.evidence_refs.append(EvidenceRef(evidence_id=rec.id))
        items = [(MemoryAtom(memcell_id=c.id, text=it["text"],
                             domains=list(it.get("domains", [])),
                             holder=it.get("holder", "user"),
                             occurrence_time=it.get("when")), it["vec"])
                 for it in atoms]
        self.cells.upsert(c, topic_embedding=np.asarray(topic_vec, dtype=np.float32)
                          if topic_vec is not None else None)
        self.atoms.upsert_many(items)
        return c

    def deps(self, embedder, reranker=None, deep_write=True, llm=None) -> DeepDeps:
        return DeepDeps(embedder=embedder, reranker=reranker or NoopReranker(),
                        atoms=self.atoms, cells=self.cells, evidence=self.ev,
                        deep_write=deep_write, llm=llm)


def _hit(cell: MemCell, atoms: list[MemoryAtom]) -> CellHit:
    return CellHit(cell=cell, score=0.0, best_sim=0.0,
                   atoms=[AtomHit(atom=a, similarity=0.5) for a in atoms])


# —— 注册表 ——

def test_registry_assigns_stable_handles():
    reg = HandleRegistry()
    assert reg.ensure("cell_a") == "c1"
    assert reg.ensure("cell_b") == "c2"
    assert reg.ensure("cell_a") == "c1"          # 重复 ensure 不换号
    assert reg.real("c2") == "cell_b"
    assert reg.real(" c1 ") == "cell_a"          # 容忍空白
    assert reg.real("c9") is None and len(reg) == 2


# —— 渲染器 ——

def test_cell_full_and_row_hide_atom_text(db):
    env = Env(db)
    c = env.add_cell(topic="画展筹备", episode="Caroline 在筹备画展,展期 2026-09。",
                     atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}],
                     evidence=[("Caroline", "展期就定九月")])
    full = cell_full(c, "c3", n_atoms=5, n_lines=2)
    assert full.splitlines()[0] == "━━━ c3 ━━━"
    assert "this segment has 5 extracted index atoms and 2 raw utterances" in full
    assert "get_cell_evidence" in full                          # 指路原话核实
    assert cell_row("c3", c) == "c3 | 2026-08-20 09:00:00 | 画展筹备"   # 目录行带秒级时间


def test_evidence_page_paginates_and_names_speaker(db):
    recs = [EvidenceRecord(holder="Caroline" if i % 2 else "user",
                           content_inline=f"第{i}句", captured_at=_T)
            for i in range(35)]
    body, total = evidence_page(recs, page=1)
    assert total == 2                                            # 35 句 → 2 页(30/页)
    assert body.count("\n") == 29                                # 首页整 30 行
    body2, _ = evidence_page(recs, page=2)
    assert "第30句" in body2 and "第34句" in body2
    assert "[2026-08-20 09:00:00] Caroline: " in body            # 说话人=holder 真名
    empty, total_e = evidence_page([], page=1)
    assert "no raw utterances kept" in empty and total_e == 1


# —— search_atoms:过滤 → MaxSim 粗排 → rerank 精排 ——

def test_search_atoms_window_filter_and_payload_shape(db):
    env = Env(db)
    # 窗内原子显式给 when:锚点若回退 recorded_at=now() 会随日历翻月掉出 8 月窗口(日历敏感修复)
    env.add_cell(topic="画展", episode="画展叙事。",
                 atoms=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0),
                         "when": datetime(2026, 8, 10, tzinfo=timezone.utc)}])
    env.add_cell(topic="跑步", episode="跑步叙事。",
                 atoms=[{"text": "跑三公里", "vec": _v(0.6, 0.8, 0, 0),
                         "when": datetime(2026, 8, 12, tzinfo=timezone.utc)}])
    env.add_cell(topic="旧画展", episode="旧叙事。",
                 atoms=[{"text": "去年画展", "vec": _v(0.95, 0.05, 0, 0),
                         "when": datetime(2026, 1, 1, tzinfo=timezone.utc)}],
                 t_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    d = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}))
    out = tool_search_atoms(d, query="展期", start_date="2026-08-01", end_date="2026-08-31")
    assert "旧叙事" not in out                                  # 日期窗挡掉 1 月的旧格
    assert "画展叙事。" in out and "跑步叙事。" in out          # 弱命中也进材料(limit 内)
    assert "展期 2026-09" not in out and "跑三公里" not in out  # atom 文本不出现
    assert "━━━ c1 ━━━" in out and "━━━ c2 ━━━" in out          # 命中格已注册编号


def test_search_atoms_rerank_reorders_candidates(db):
    env = Env(db)
    strong = env.add_cell(topic="画展", episode="画展叙事。",
                          atoms=[{"text": "展期", "vec": _v(1, 0, 0, 0)}])
    weak = env.add_cell(topic="跑步", episode="跑步叙事。",
                        atoms=[{"text": "配速", "vec": _v(0.6, 0.8, 0, 0)}])
    d = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}))
    noop_out = tool_search_atoms(d, query="展期")
    assert noop_out.index("画展叙事。") < noop_out.index("跑步叙事。")   # Noop:MaxSim 序

    class FlipReranker:   # 精排强行倒序,证明 rerank 真的接管了最终序
        def rerank(self, query, documents, *, instruction=""):
            return list(reversed([0.9 / (i + 1) for i in range(len(documents))]))

    d2 = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}), reranker=FlipReranker())
    flip_out = tool_search_atoms(d2, query="展期")
    assert flip_out.index("跑步叙事。") < flip_out.index("画展叙事。")   # 精排序生效


def test_search_atoms_empty_gives_actionable_advice(db):
    env = Env(db)
    env.add_cell(topic="画展", episode="叙事。",
                 atoms=[{"text": "展期", "vec": _v(1, 0, 0, 0), "domains": ["D13"]}])
    d = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}))
    out = tool_search_atoms(d, query="展期", domains=["D99"])    # 域过滤清空池
    assert "No hits" in out and "find_cells" in out


def test_search_atoms_limit_clamped_to_8(db):
    env = Env(db)
    for i in range(12):
        env.add_cell(topic=f"t{i}", atoms=[{"text": f"x{i}", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(TableEmbedder({"q": _v(1, 0, 0, 0)}))
    out = tool_search_atoms(d, query="q", limit=99)
    assert "search_atoms hit 8 unit(s)" in out                  # 硬上限 8


# —— search_atoms 链扩展(S5,§6)——

def _mk_chain(db, env, title, specs):
    """按 (cell_id, text) 取库内真实 atom 建链(直接 ChainStore,不走判链)。"""
    cs = ChainStore(db)
    members = [next(a for a in env.atoms.list_by_cell(cid) if a.text == t)
               for cid, t in specs]
    info = ChainInfo(title=title)
    cs.create_chain(info, members[0], centroid=None)
    for m in members[1:]:
        cs.append_atom(info, m, centroid=None)
    return cs.get_chain(info.id)


class _StubLLM:
    """织写器替身:固定返回织文(chat 签名与 ChatLLM 协议一致)。"""
    def __init__(self, text="织文:馆先在健身房,后搬到MBS。"):
        self.text = text
    def chat(self, messages, temperature=0.3, max_tokens=2048):
        return self.text


def test_search_atoms_weaves_chain_members(db, monkeypatch):
    """命中 ≥2 节点链 → 与快链同机制织成 memcell′:织文替换散格,块头列成员格 handle,
    成员格全部注册可下钻;atom 文本不出现,无关格不被带出。"""
    env = Env(db)
    c1 = env.add_cell(topic="瑜伽一", episode="瑜伽叙事一。",
                      atoms=[{"text": "馆在健身房", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="跑步", episode="跑步叙事。",
                 atoms=[{"text": "跑三公里", "vec": _v(0.6, 0.8, 0, 0)}])
    c3 = env.add_cell(topic="瑜伽二", episode="瑜伽叙事二。",
                      atoms=[{"text": "馆搬到了MBS", "vec": _v(0.5, 0.86, 0, 0)}])
    _mk_chain(db, env, "用户瑜伽地点", [(c1.id, "馆在健身房"), (c3.id, "馆搬到了MBS")])

    d = env.deps(TableEmbedder({"瑜伽": _v(1, 0, 0, 0)}), llm=_StubLLM())
    out = tool_search_atoms(d, query="瑜伽", limit=2)
    assert "织文:馆先在健身房" in out                            # 织写单元(合并叙事)
    assert "瑜伽叙事一。" not in out and "瑜伽叙事二。" not in out  # 散格被织文替换
    assert "woven from fact-chains" in out
    h1, h3 = d.reg.real("c1"), d.reg.real("c2")
    assert {h1, h3} == {c1.id, c3.id}                           # 成员格都注册了(可 open 下钻)
    assert f"━━━ c1, c2 ━━━" in out                             # 织写单元块头多 handle 并列
    assert "跑步叙事。" in out                                  # 普通单元并存
    assert "(pulled via chain" not in out                       # 旧散格扩展已删除


def test_search_atoms_weave_covers_whole_long_chain(db, monkeypatch):
    """长链无截断:8 成员链命中 → 织写单元覆盖全部 8 格(旧实现封顶 5 格且静默截断)。"""
    env = Env(db)
    specs = []
    for i in range(8):
        day = datetime(2026, 8, 1 + i, tzinfo=timezone.utc)
        c = env.add_cell(topic=f"格{i}", episode=f"第{i}话",
                         t_start=day,
                         atoms=[{"text": f"事实{i}",
                                 "vec": _v(1, 0, 0, 0) if i == 0 else _v(0.3, 0.95, 0, 0),
                                 "when": day}])
        specs.append((c.id, f"事实{i}"))
    _mk_chain(db, env, "长链", specs)
    d = env.deps(TableEmbedder({"瑜伽": _v(1, 0, 0, 0)}), llm=_StubLLM("织文:长链全史。"))
    out = tool_search_atoms(d, query="瑜伽", limit=1)
    assert "织文:长链全史。" in out
    handles = [d.reg.ensure(c.id) for c in env.cells.iter_all()]  # 应已注册 8 格
    assert len(handles) == 8
    assert "c1, c2, c3, c4, c5, c6, c7, c8" in out               # 块头列出全部成员格


def test_search_atoms_chain_degrades_without_llm(db, monkeypatch):
    """无 llm(测试/降级):链成员退回普通格单元,机制与快链降级路径一致。"""
    env = Env(db)
    c1 = env.add_cell(topic="瑜伽一", episode="瑜伽叙事一。",
                      atoms=[{"text": "馆在健身房", "vec": _v(1, 0, 0, 0)}])
    c3 = env.add_cell(topic="瑜伽二", episode="瑜伽叙事二。",
                      atoms=[{"text": "馆搬到了MBS", "vec": _v(0.5, 0.86, 0, 0)}])
    _mk_chain(db, env, "用户瑜伽地点", [(c1.id, "馆在健身房"), (c3.id, "馆搬到了MBS")])
    d = env.deps(TableEmbedder({"瑜伽": _v(1, 0, 0, 0)}))         # llm=None
    out = tool_search_atoms(d, query="瑜伽", limit=2)
    assert "瑜伽叙事一。" in out and "瑜伽叙事二。" in out        # 成员格作普通单元
    assert "woven" not in out


# —— find_cells:时间窗/域过滤 + topic 相似度 + 分页 ——

def test_find_cells_domain_filter_and_time_desc(db):
    env = Env(db)
    aug = env.add_cell(topic="画展", domains=["D13"],
                       t_start=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc))
    jul = env.add_cell(topic="旅行", domains=["D13"],
                       t_start=datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc))
    env.add_cell(topic="跑步", domains=["D05"],
                 t_start=datetime(2026, 8, 1, 7, 0, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    out = tool_find_cells(d, domains=["D13"])
    rows = [ln for ln in out.splitlines() if ln.startswith("c")]
    assert [d.reg.real(ln.split(" |")[0]) for ln in rows] == [aug.id, jul.id]   # 时间倒序,域外剔除
    assert "跑步" not in out


def test_find_cells_window_inclusive_of_end_day(db):
    env = Env(db)
    in_c = env.add_cell(topic="画展", t_start=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc))
    env.add_cell(topic="旅行", t_start=datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc))
    edge = env.add_cell(topic="跑步", t_start=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    out = tool_find_cells(d, start_date="2026-08-01", end_date="2026-08-31")
    assert "旅行" not in out                                    # 窗外剔除
    ids = {d.reg.real(ln.split(" |")[0]) for ln in out.splitlines() if ln.startswith("c")}
    assert ids == {in_c.id, edge.id}                            # 含端点当天


def test_find_cells_query_ranks_by_topic_similarity(db):
    env = Env(db)
    env.add_cell(topic="跑步计划", topic_vec=_v(0.2, 0.98))
    near = env.add_cell(topic="画展筹备", topic_vec=_v(0.98, 0.2))
    d = env.deps(TableEmbedder({"画展": _v(1, 0)}))
    out = tool_find_cells(d, query="画展")
    assert "topic similarity descending" in out
    first = out.splitlines()[1].split(" |")[0]
    assert d.reg.real(first) == near.id                         # topic 近的排前


def test_find_cells_paging_tail(db):
    env = Env(db)
    for i in range(13):
        env.add_cell(topic=f"t{i:02d}", t_start=datetime(2026, 8, i + 1, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    p1 = tool_find_cells(d, page=1)
    assert "find_cells page 1/2 (13 cell(s) after filtering" in p1 and "page=2" in p1
    assert "t12" in p1 and "t00" not in p1                   # 倒序:最新在前
    p2 = tool_find_cells(d, page=2)
    assert "(last page)" in p2 and "t00" in p2 and "t12" not in p2


# —— open_cell / get_cell_evidence ——

def test_open_cell_unknown_and_known(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。",
                     atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(ZeroEmbedder())
    assert "Unknown handle" in tool_open_cell(d, c="c9")
    d.reg.ensure(c.id)
    out = tool_open_cell(d, c="c1")
    assert "叙事全文。" in out and "1 extracted index atoms" in out


def test_get_cell_evidence_renders_by_speaker(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事。",
                     evidence=[("user", "你说展期九月?"), ("Caroline", "对,2026-09-01 开展")])
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    out = tool_get_cell_evidence(d, c="c1")
    assert "2 utterance(s), page 1/1" in out
    assert "user: 你说展期九月?" in out and "Caroline: 对,2026-09-01 开展" in out
    assert "Unknown handle" in tool_get_cell_evidence(d, c="c8")


# —— search_evidence:关键词直搜原话(兜底路径,不经索引)——

def test_search_evidence_finds_utterance_atoms_missed(db):
    """兜底场景:事实只存在原话里(atoms 零抽取)也能搜到,并给出所属格编号。"""
    env = Env(db)
    c = env.add_cell(topic="厨房琐事", episode="聊了些家里的事。",
                     atoms=(),                                    # 抽取漏了,索引为空
                     evidence=[("Melanie", "I broke my favourite bowl last night"),
                               ("user", "没事,再买一个就好")])
    d = env.deps(ZeroEmbedder())
    out = tool_search_evidence(d, keywords=["bowl", "broke"])
    assert "search_evidence hit 1 utterance(s)" in out
    assert "Melanie: I broke my favourite bowl last night" in out
    assert "(cell c1)" in out


def test_search_evidence_requires_all_keywords_same_sentence(db):
    """关键词 AND 同句:两句各含一个词不算命中。"""
    env = Env(db)
    env.add_cell(topic="t", episode="e",
                 evidence=[("user", "the bowl is blue"), ("user", "I broke my pen")])
    d = env.deps(ZeroEmbedder())
    assert "No hits in the raw transcript either" in tool_search_evidence(d, keywords=["bowl", "broke"])


def test_search_evidence_holder_filter_and_guards(db):
    env = Env(db)
    env.add_cell(topic="t", episode="e",
                 evidence=[("user", "bowl broke"), ("Melanie", "bowl broke too")])
    d = env.deps(ZeroEmbedder())
    out = tool_search_evidence(d, keywords=["bowl"], holder="Melanie")
    assert "Melanie:" in out and "user:" not in out
    assert "keywords is empty" in tool_search_evidence(d, keywords=["", " "])


def test_search_evidence_schema_tolerates_junk_limit():
    from personos.online.deep_recall import _SearchEvidenceArgs
    args = _SearchEvidenceArgs(keywords=["x"], limit="abc")
    assert args.limit == 15                                       # 坏值回落默认


# —— remember:只新增 + quote 回链 + 护栏 ——

def _read_emb(env, atom_id) -> bytes:
    row = env.atoms.db.fetch_one("SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s",
                                 (atom_id,))
    return bytes.fromhex(row["emb"])


def test_remember_writes_atom_with_refs_and_appends_episode(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="原叙事。",
                     evidence=[("Caroline", "展期包含儿童展区")])
    stamped = "[2026-08-20] Caroline 说画展新增儿童展区"          # stamped_atom_text 的产物
    d = env.deps(TableEmbedder({stamped: _v(1, 0, 0, 0)}))
    d.reg.ensure(c.id)
    out = tool_remember(d, c="c1", text="Caroline 说画展新增儿童展区",
                        quote="展期包含儿童展区", holder="Caroline", kind="K04",
                        domains=["D13"], episode_append="补充:新增了儿童展区。")
    assert "Written back:" in out and "added 1 index atom (linked to 1 source utterance(s))" in out
    saved = env.atoms.list_by_cell(c.id)
    assert len(saved) == 1 and saved[0].source == "deep"        # 来源标记
    assert saved[0].kind == "K04" and saved[0].holder == "Caroline"
    assert saved[0].evidence_refs                               # quote 逐字回链到证据
    assert np.frombuffer(_read_emb(env, saved[0].id),
                         dtype=np.float32).tolist() == [1, 0, 0, 0]   # 用 stamped 文本 embed
    episode = env.cells.get(c.id).episode
    assert episode.startswith("原叙事。") and episode.endswith("补充:新增了儿童展区。")  # 追加不改写


def test_remember_quote_miss_leaves_refs_empty_but_writes(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="原叙事。",
                     evidence=[("Caroline", "展期包含儿童展区")])
    d = env.deps(TableEmbedder({"[2026-08-20] x": _v(1, 0, 0, 0)}))
    d.reg.ensure(c.id)
    out = tool_remember(d, c="c1", text="x", quote="原话里没有这句")   # 未逐字命中
    assert "added 1 index atom (linked to 0 source utterance(s))" in out
    assert env.atoms.list_by_cell(c.id)[0].evidence_refs == []


def test_remember_guards(db):
    env = Env(db)
    c = env.add_cell(topic="t", episode="e")
    d_off = env.deps(ZeroEmbedder(), deep_write=False)
    d_off.reg.ensure(c.id)
    assert "read-only" in tool_remember(d_off, c="c1", text="x")   # 总开关

    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    assert "at least one of text / episode_append" in tool_remember(d, c="c1")   # 空写
    assert "Cannot write back" in tool_remember(d, c="c7", text="x")   # 未知编号
    d.remembered = 8
    assert "Write-back cap" in tool_remember(d, c="c1", text="x")   # 会话上限


# —— MaasChatModel:消息映射 + stop 客户端截断 ——

def test_maas_chat_model_maps_roles_and_truncates_at_stop():
    captured: list[list[dict]] = []

    class SpyLLM:
        def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
            captured.append({"t": temperature, "m": max_tokens, "msgs": list(messages)})
            return "前半\nObservation:幻觉续写"

    from langchain_core.messages import HumanMessage, SystemMessage
    m = MaasChatModel(client=SpyLLM(), temperature=0.1, max_tokens=64)
    res = m.invoke([SystemMessage(content="系统"), HumanMessage(content="问")],
                   stop=["\nObservation"])
    assert res.content == "前半"                                 # stop 客户端截断(防幻觉续写)
    assert captured[0]["msgs"] == [{"role": "system", "content": "系统"},
                                   {"role": "user", "content": "问"}]   # role 映射
    assert captured[0]["t"] == 0.1 and captured[0]["m"] == 64    # 参数透传


# —— run_deep:agent 循环(交接包 → 工具 → 终答回译)——

def test_open_cell_accepts_common_arg_aliases(db):
    """真跑踩坑:模型常把 c 写成 cell_id/cell——schema 别名兜住,不让整轮 agent 崩。"""
    from personos.online.deep_recall import build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。",
                     atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    tools = {t.name: t for t in build_tools(d)}
    assert "叙事全文。" in tools["open_cell"].invoke({"cell_id": "c1"})
    assert "0 utterance(s)" in tools["get_cell_evidence"].invoke({"cell": "c1"})
    # 真跑踩坑:模型照描述文案 "Cell handle" 猜参数名 handle/cell_handle——
    # 别名不收就每题浪费一步试错(先撞校验失败再自纠)。
    assert "叙事全文。" in tools["open_cell"].invoke({"handle": "c1"})
    assert "0 utterance(s)" in tools["get_cell_evidence"].invoke({"cell_handle": "c1"})


def test_tool_batch_runs_many_calls_in_one_step(db):
    """action_input 传参数对象列表 = 一步批量调用:逐个执行,只算一步预算。"""
    from personos.online.deep_recall import _MAX_STEPS, build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。",
                     atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    tools = {t.name: t for t in build_tools(d)}
    out = tools["open_cell"].invoke({"batch": [{"c": "c1"}, {"c": "c1"}]})
    assert "── batch 1/2 ──" in out and "── batch 2/2 ──" in out
    assert out.count("叙事全文。") == 2                          # 两次调用各自执行
    assert d.calls_made == 1                                    # 批量只烧一步
    assert f"≈{_MAX_STEPS - 1} tool steps left" in out          # 观察尾部附剩余预算


def test_tool_budget_notice_presses_for_answer_near_cap(db):
    """预算剩 ≤2 步时,观察尾部升级为"立即收手作答"提示。"""
    from personos.online.deep_recall import _MAX_STEPS, build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。")
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    d.calls_made = _MAX_STEPS - 3                             # 本次调用后剩 2 步
    tools = {t.name: t for t in build_tools(d)}
    out = tools["open_cell"].invoke({"c": "c1"})
    assert "≈2 tool steps left" in out and "budget nearly exhausted" in out


def test_search_tools_accept_singular_aliases(db):
    """真跑踩坑:模型爱用单数 keyword/domain——别名兜住,不再烧一步试错。"""
    from personos.online.deep_recall import build_tools
    env = Env(db)
    env.add_cell(topic="画展", episode="叙事全文。",
                 atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}],
                 evidence=[("user", "周末去看画展")])
    d = env.deps(ZeroEmbedder())
    tools = {t.name: t for t in build_tools(d)}
    out = tools["search_evidence"].invoke({"keyword": "画展"})   # 单数 + 裸字符串
    assert "周末去看画展" in out
    out = tools["find_cells"].invoke({"domain": ["D01"]})
    assert "Tool argument validation failed" not in out
    out = tools["search_atoms"].invoke({"query": "x", "domain": ["D99"]})
    assert "Tool argument validation failed" not in out


def test_schema_tolerates_junk_scalar_args(db):
    """真跑踩坑:模型偶发把 page 传成 "abc"——schema 入参层容错回落默认值,循环不崩。"""
    env = Env(db)
    env.add_cell(topic="t", episode="e")
    llm = FakeLLM([
        '```json\n{"thought": "乱来", "action": "find_cells", "action_input": {"page": "abc"}}\n```',
        '```json\n{"thought": "收手", "action": "Final Answer",'
        ' "action_input": {"answer": "答", "cited": []}}\n```',
    ])
    out = run_deep(llm, ZeroEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    assert out.ans.answer == "答" and len(out.steps) == 1
    assert "t" in out.steps[0]["obs_head"]               # 垃圾 page 回落 1,工具正常返回


def test_tool_exception_becomes_observation_not_crash(db):
    """handle_tool_error:工具函数内异常(embedder 挂)→ 观察文本,agent 循环不中断。"""
    class BoomEmbedder:
        def embed(self, _texts):
            raise RuntimeError("embed 服务不可用")

    env = Env(db)
    env.add_cell(topic="t", episode="e", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])   # pool 非空才会走到 embed
    llm = FakeLLM([
        '```json\n{"thought": "先检索", "action": "search_atoms", "action_input": {"query": "q"}}\n```',
        '```json\n{"thought": "检索挂了,直接作答", "action": "Final Answer",'
        ' "action_input": {"answer": "答", "cited": []}}\n```',
    ])
    out = run_deep(llm, BoomEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    assert out.ans.answer == "答" and len(out.steps) == 1
    assert "embed 服务不可用" in out.steps[0]["obs_head"]  # 异常变成观察,agent 可自纠


def test_wrong_field_name_becomes_observation_not_crash(db):
    """字段名彻底写错(target≠c/别名)→ 校验错误变观察提示,agent 循环不中断。"""
    env = Env(db)
    env.add_cell(topic="t", episode="e")
    llm = FakeLLM([
        '```json\n{"thought": "展开", "action": "open_cell", "action_input": {"target": "c1"}}\n```',
        '```json\n{"thought": "改对字段名重来", "action": "open_cell", "action_input": {"c": "c1"}}\n```',
        '```json\n{"thought": "齐了", "action": "Final Answer",'
        ' "action_input": {"answer": "答", "cited": []}}\n```',
    ])
    out = run_deep(llm, ZeroEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    assert out.ans.answer == "答" and len(out.steps) == 2
    assert "Tool argument validation failed" in out.steps[0]["obs_head"]
    assert "c1" in out.steps[1]["obs_head"] and "\ne\n" in out.steps[1]["obs_head"]


def test_run_deep_agent_loop_and_citation(db):
    env = Env(db)
    c = env.add_cell(topic="画展筹备", episode="Caroline 在筹备画展,展期 2026-09。",
                     atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}],
                     evidence=[("Caroline", "展期就定九月")])
    llm = FakeLLM([
        '```json\n{"thought": "先按事实检索", "action": "search_atoms",'
        ' "action_input": {"query": "画展 展期", "limit": 3}}\n```',
        '```json\n{"thought": "材料已足够", "action": "Final Answer",'
        ' "action_input": {"answer": "展期在 2026-09。", "cited": ["c1", "c9"]}}\n```',
    ])
    out = run_deep(llm, TableEmbedder({"画展 展期": _v(1, 0, 0, 0)}),
                   env.atoms, env.cells, env.ev,
                   query="画展什么时候", now_dt=_T, deep_write=False)
    assert out.ans.answer == "展期在 2026-09。"
    assert out.ans.cited_cells == [c.id]                        # c9 未知 → 丢弃,c1 回译真 id
    assert [(s["tool"], s["args"]["query"]) for s in out.steps] == [("search_atoms", "画展 展期")]
    for part in ("## Task", "## Memory catalog", "## Current time"):   # 交接包六件套在场
        assert part in out.handoff
    assert out.secs["deep"] >= 0


def test_run_deep_iteration_cap_yields_empty_answer(db):
    env = Env(db)
    env.add_cell(topic="t", episode="e", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    loop = ('```json\n{"thought": "再翻一页", "action": "open_cell", "action_input": {"c": "c1"}}\n```')
    llm = FakeLLM([loop] * 20)                                  # 永远不停 → 触发 9 步上限
    out = run_deep(llm, ZeroEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    assert out.ans.answer == "" and len(out.steps) == 9         # 如实"没答出来",不硬凑


def test_handoff_carries_parts_without_answer_draft(db):
    env = Env(db)
    c1 = env.add_cell(topic="画展", episode="画展叙事。",
                      t_start=datetime(2026, 8, 20, tzinfo=timezone.utc))
    c2 = env.add_cell(topic="跑步", episode="跑步叙事。",
                      t_start=datetime(2026, 8, 10, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    from personos.online.arbitrate import ReviewResult
    review = ReviewResult(verdict="insufficient_material",
                          critique="缺画展的具体展期日期(建议从画展相关格找起)")
    handoff = build_handoff(d, query="画展展期?", now_dt=_T,
                            rw=QueryRewrite(original="画展展期?", resolved="画展展期?",
                                            subject="Caroline", time_start="2026-08-01",
                                            time_end="2026-08-31", domains=["D13"]),
                            review=review, fast_hits=[_hit(c1, env.atoms.list_by_cell(c1.id))],
                            total_cells=2, catalog=[c1, c2])
    assert "画展展期?" in handoff and "question subject: Caroline" in handoff   # ①任务
    assert "2026-08-01 ~ 2026-08-31" in handoff and "D13" in handoff
    assert "画展叙事。" in handoff and "━━━ c1 ━━━" in handoff        # ②快链证据全文
    assert "the retrieved materials lack" in handoff and "缺画展的具体展期日期" in handoff   # ③核判+缺口原样
    assert "跑步" in handoff and "0 older cells exist" in handoff        # ④目录
    assert "## Current time" in handoff                                 # ⑤当前时间
    assert "作答" not in handoff or "草稿" not in handoff              # 不给作答草稿


def test_handoff_translates_fast_window_handles_to_deep_handles(db):
    """两套编号各自独立:核判 critique 写快链 mN(材料窗口序),交接时回译成深轨 cN
    (目录注册序,c1=最新)——材料区用深轨编号渲染,缺口引用必须与材料区对得上号。
    越界编号原样保留:可见的陌生符号,而不是静默指错格。"""
    from personos.online.arbitrate import ReviewResult
    env = Env(db)
    T = datetime(2026, 8, 27, 21, 0, tzinfo=timezone.utc)
    prep = env.add_cell(topic="筹备画展", episode="筹备叙事。", t_start=T)
    venue = env.add_cell(topic="画展场地", episode="场地叙事。", t_start=T)
    sophia = env.add_cell(topic="Sophia 扭伤", episode="扭伤叙事。", t_start=T)
    d = env.deps(ZeroEmbedder())
    review = ReviewResult(verdict="answer_defect",
                          critique="materials contain 2 venue facts (m1, m2); m9 is bogus")
    handoff = build_handoff(d, query="画展场地在哪?", now_dt=_T, rw=None, review=review,
                            fast_hits=[_hit(venue, []), _hit(prep, [])],
                            total_cells=3, catalog=[sophia, venue, prep])
    assert "━━━ c2 ━━━" in handoff and "场地叙事。" in handoff    # 深轨编号:venue=c2(目录序)
    assert "(c2, c3)" in handoff          # m1→venue=c2、m2→prep=c3:与材料区同一套号
    assert "m1" not in handoff and "m2" not in handoff
    assert "m9 is bogus" in handoff       # 越界:原样保留,不臆造映射


# —— run_recall 分岔(编排层断言)——

def test_run_recall_deep_mode_bypasses_fast(db, evidence_store):
    from personos.online.recall_flow import run_recall
    env = Env(db)
    c = env.add_cell(topic="画展", episode="画展叙事。",
                     atoms=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0), "展期": _v(1, 0, 0, 0)})
    state = {"agent": 0}

    class DeepLLM:
        """R0 走查询预处理器;深轨系统词命中时回 agent JSON 块;其余工位=不该被调。"""

        def chat(self, messages, temperature=0.3, max_tokens=2048):
            sys = messages[0]["content"]
            if "query preprocessor" in sys:
                return ('{"resolved":"画展筹备","subject":"","expansions":[],'
                        '"time_start":null,"time_end":null,"domains":[]}')
            if "deep-retrieval agent" in sys:
                state["agent"] += 1
                if state["agent"] == 1:
                    return ('```json\n{"thought": "检索", "action": "search_atoms",'
                            ' "action_input": {"query": "展期"}}\n```')
                return ('```json\n{"thought": "够了", "action": "Final Answer",'
                        ' "action_input": {"answer": "展期在 2026-09。", "cited": ["c1"]}}\n```')
            raise AssertionError(f"deep 模式不该调这个工位: {sys[:40]!r}")

    o = run_recall(DeepLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="deep")
    assert o.hits == [] and o.reviews == []                      # 快链工位全跳过
    assert o.deep is not None and o.ans.answer == "展期在 2026-09。"
    assert o.ans.cited_cells == [c.id] and state["agent"] == 2
    assert not o.escalated and "deep" in o.secs


def test_run_recall_auto_insufficient_escalates_and_deep_overrides(db, evidence_store):
    from personos.online.recall_flow import run_recall
    env = Env(db)
    c = env.add_cell(topic="画展", episode="画展叙事。",
                     atoms=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0), "展期": _v(1, 0, 0, 0)})

    class EscalateLLM:
        def chat(self, messages, temperature=0.3, max_tokens=2048):
            sys = messages[0]["content"]
            if "query preprocessor" in sys:
                return ('{"resolved":"画展筹备","subject":"","expansions":[],'
                        '"time_start":null,"time_end":null,"domains":[]}')
            if "answer reviewer" in sys:      # 核判词先于 answerer(核判 prompt 里也出现该词)
                return '{"verdict":"insufficient_material","critique":"缺展期日期"}'
            if "answerer" in sys:
                return '{"answer":"(快链的凑合答案)","cells":["c1"]}'
            if "deep-retrieval agent" in sys:
                return ('```json\n{"thought": "直接终答", "action": "Final Answer",'
                        ' "action_input": {"answer": "展期在 2026-09。", "cited": ["c1"]}}\n```')
            raise AssertionError(f"未知工位: {sys[:40]!r}")

    o = run_recall(EscalateLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="auto")
    assert o.escalated and o.deep is not None
    assert not o.retried                                        # 材料不足不重答,直升深轨
    assert "缺展期日期" in o.deep.handoff                        # 指正当缺口方向交给深轨
    assert o.ans.answer == "展期在 2026-09。"                    # 深轨终答覆盖快链
    assert o.ans.cited_cells == [c.id]


def test_run_recall_auto_ok_does_not_escalate(db, evidence_store):
    from personos.online.recall_flow import run_recall
    env = Env(db)
    env.add_cell(topic="画展", episode="画展叙事。",
                 atoms=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0)})

    class OkLLM:
        def chat(self, messages, temperature=0.3, max_tokens=2048):
            sys = messages[0]["content"]
            if "query preprocessor" in sys:
                return ('{"resolved":"画展筹备","subject":"","expansions":[],'
                        '"time_start":null,"time_end":null,"domains":[]}')
            if "answer reviewer" in sys:      # 核判词先于 answerer(核判 prompt 里也出现该词)
                return '{"verdict":"ok","critique":""}'
            if "answerer" in sys:
                return '{"answer":"展期在 2026-09。","cells":["c1"]}'
            raise AssertionError(f"ok 不该升深轨: {sys[:40]!r}")

    o = run_recall(OkLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="auto")
    assert not o.escalated and o.deep is None and not o.retried
    assert o.ans.answer == "展期在 2026-09。"
