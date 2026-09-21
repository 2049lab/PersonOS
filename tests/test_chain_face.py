"""单元组装测试(docs/atom-chain-design.md §5.2/§5.3):分桶/去重/织写与降级/残缺提示。

零真实 MAAS:组装零 LLM(单节点链/游离);织写用替身(RecorderLLM 断言 prompt 契约,
FakeLLM callable 区分多链,BoomLLM 断言降级)。
"""

from __future__ import annotations

from datetime import datetime, timezone

from personos.models import ChainInfo, MemCell, MemoryAtom
from personos.online.chain_face import assemble_units
from personos.online.retrieval import AtomHit, CellHit, answer_from_cells, cell_block
from personos.storage.chain_store import ChainStore

from .fakes import FakeLLM
from .test_retrieval import Env, _v

_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


class RecorderLLM:
    """记录全部 messages 的替身:织写 prompt 契约断言用(system+user 都要验)。"""

    def __init__(self, resp: str):
        self.resp = resp
        self.calls: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        self.calls.append(messages)
        return self.resp


class CountingLLM:
    """只数调用次数:断言单节点链/纯游离不调 LLM。"""

    calls = 0

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        CountingLLM.calls += 1
        return "织文"


class BoomLLM:
    """chat 必炸:断言单链织写失败只降级该链。"""

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        raise RuntimeError("maas down")


def _pool_of(env, items: list[tuple[str, str]]):
    """按 (cell_id, text) 池序取库内真实 atom 拼 AtomHit 池(相似度/rrf 递减)。"""
    out = []
    for i, (cid, text) in enumerate(items):
        a = next(x for x in env.atoms.list_by_cell(cid) if x.text == text)
        out.append(AtomHit(atom=a, similarity=0.9 - i * 0.01, rrf=0.03 - i * 0.001))
    return out


def _chain(db, env, title: str, specs: list[tuple[str, str]]):
    """按 (cell_id, text) 拿库内真实 atom 建链(直接 ChainStore,不走判链)。返回最新 ChainInfo。"""
    cs = ChainStore(db)
    members = [next(a for a in env.atoms.list_by_cell(cid) if a.text == t)
               for cid, t in specs]
    info = ChainInfo(title=title)
    cs.create_chain(info, members[0], centroid=None)
    for m in members[1:]:
        cs.append_atom(info, m, centroid=None)
    return cs.get_chain(info.id)


# —— 普通单元(无链库的通用形态)——

def test_plain_units_no_chains(db):
    """无链库:材料 = 池 atom 所在 cell 去重(首现位序);零 LLM。"""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(33)]

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    units = asm.units

    assert asm.n_pool == 33 and asm.n_chains == 0 and asm.n_woven == 0
    assert [u.cell.id for u in units] == ids                    # 首现位序 + cell 去重
    assert all(isinstance(u, CellHit) and not u.covers for u in units)
    assert asm.boundary == ""                                   # 无池外 atom → 无提示


def test_same_cell_multiple_atoms_dedup(db):
    """一格多 atom 进池 → 一个普通单元(cell 去重);单元 atoms 按相似度降序。"""
    env = Env(db)
    c1 = env.add_cell(topic="一格多 atom", episode="叙事",
                      atoms=[{"text": "a", "vec": _v(1, 0, 0, 0)},
                             {"text": "b", "vec": _v(0.99, 0.1, 0, 0)},
                             {"text": "c", "vec": _v(0.98, 0.2, 0, 0)}])
    c2 = env.add_cell(topic="另一格", episode="叙事二", atoms=[{"text": "d", "vec": _v(0.9, 0.4, 0, 0)}])

    pool = _pool_of(env, [(c1.id, "c"), (c2.id, "d"), (c1.id, "a"), (c1.id, "b")])
    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    assert asm.n_pool == 4
    assert [u.cell.id for u in asm.units] == [c1.id, c2.id]     # c1 三个 atom 只出一单元
    # 单元 atoms 按相似度降序:池序 c(0.90)>a(0.87)>b(0.86)(d 不属于 c1)
    assert [ah.atom.text for ah in asm.units[0].atoms] == ["c", "a", "b"]


def test_orphan_cell_skipped(db):
    """cell 已不存在的孤儿 atom:跳过不崩,不产空壳单元。"""
    env = Env(db)
    c = env.add_cell(topic="健在", episode="叙事",
                     atoms=[{"text": "好", "vec": _v(1, 0, 0, 0)}])
    env.atoms.upsert(MemoryAtom(memcell_id="cell_ghost", text="孤儿"), )
    ghost = next(a for a in env.atoms.list(limit=99) if a.text == "孤儿")
    pool = [AtomHit(atom=next(a for a in env.atoms.list_by_cell(c.id)), similarity=0.9, rrf=0.03),
            AtomHit(atom=ghost, similarity=0.8, rrf=0.02)]

    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    assert [u.cell.id for u in asm.units] == [c.id]             # 孤儿跳过,健在格保留


# —— 链映射:单节点零 LLM,≥2 节点织写 ——


def test_single_node_chain_is_plain_no_llm(db):
    """单节点链 = 它的 memcell 普通单元,零 LLM;悬空 chain_id 同样按游离处理。"""
    env = Env(db)
    c4 = env.add_cell(topic="独链", episode="叙事四",
                      atoms=[{"text": "单条事实", "vec": _v(0.93, 0.36, 0, 0)}])
    _chain(db, env, "单 atom 链", [(c4.id, "单条事实")])     # n_atoms=1 → 普通单元语义

    CountingLLM.calls = 0
    asm = assemble_units(_pool_of(env, [(c4.id, "单条事实")]),
                         ChainStore(db), env.cells, CountingLLM(), query="q")
    assert CountingLLM.calls == 0                              # 单节点链不调 LLM
    assert [u.cell.id for u in asm.units] == [c4.id] and not asm.units[0].covers


def test_mixed_cell_chain_and_plain_coexist(db):
    """同格既有链内 atom 又有游离 atom:织写单元与该格普通单元并存(链事实织入,格叙事仍全)。"""
    env = Env(db)
    c1 = env.add_cell(topic="混合格", episode="叙事一",
                      atoms=[{"text": "链内事实", "vec": _v(1, 0, 0, 0)},
                             {"text": "游离事实", "vec": _v(0.97, 0.2, 0, 0)}])
    c2 = env.add_cell(topic="链友格", episode="叙事二",
                      atoms=[{"text": "链内事实二", "vec": _v(0.96, 0.28, 0, 0)}])
    ch = _chain(db, env, "一条链", [(c1.id, "链内事实"), (c2.id, "链内事实二")])   # 游离 atom 不挂链

    asm = assemble_units(_pool_of(env, [(c1.id, "链内事实"), (c1.id, "游离事实"), (c2.id, "链内事实二")]),
                         ChainStore(db), env.cells, FakeLLM(["织文。"]), query="q")
    assert asm.n_woven == 1 and len(asm.units) == 2            # 织写单元 + c1 普通单元(c2 被链消费)
    woven, plain = asm.units
    assert woven.cell.id == ch.id and plain.cell.id == c1.id
    assert [ah.atom.text for ah in plain.atoms] == ["游离事实"]  # 普通单元只带游离 atom


# —— 织写单元语义(covers 引用展开;mN 泛化)——

def test_woven_unit_citation_expansion():
    """织写单元(covers=成员格)被引 mN → cited 展开为全部成员格 id,跨单元去重。"""
    cc = MemCell(id="cell_c", topic="C")
    woven = CellHit(cell=MemCell(id="chn_1", topic="链题"), score=2.0, best_sim=0.9,
                    covers=["cell_a", "cell_b"])
    plain = CellHit(cell=cc, score=1.0, best_sim=0.8)
    llm = FakeLLM(['{"answer":"织格+普通格。","cells":["m1","m2","m1"]}'])

    ans = answer_from_cells(llm, query="q", subject="", hits=[woven, plain])
    assert ans.cited_cells == ["cell_a", "cell_b", "cell_c"]   # m1 展开,m1 重复引用不重复计


def test_cell_block_uniform_lead():
    """统一渲染:织写 memcell′ 与普通格同一材料头格式(topic 位=链title,时间=跨度)。"""
    c = MemCell(id="chn_1", topic="链题", episode="织文。",
                t_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                t_end=datetime(2026, 8, 30, tzinfo=timezone.utc))
    block = cell_block(CellHit(cell=c, score=1.0, best_sim=0.9, covers=["cell_a"]), "m1")
    assert "━━━ m1 ━━━" in block
    assert "[dialogue 2026-08-01 to 2026-08-30 | topic: 链题]" in block   # 与普通格同格式
    assert "[woven" not in block and "chain:" not in block                # 无特殊头
    plain = cell_block(CellHit(cell=MemCell(id="cell_b", topic="题", episode="叙"), score=1.0,
                               best_sim=0.9), "m2")
    assert "[dialogue date unknown | topic: 题]" in plain


# —— 织写器(S4,§5.3)——

def _yoga_scene(db):
    """三格场景:c1/c3 同链(瑜伽地点),c2 游离(跑步)。返回 (env, c1, c2, c3)。"""
    env = Env(db)
    d1, d3 = datetime(2026, 8, 10, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)
    c1 = env.add_cell(topic="瑜伽一", episode="叙事一",
                      atoms=[{"text": "馆在 MBS", "vec": _v(1, 0, 0, 0), "when": d1}])
    c1.t_start = c1.t_end = d1                      # add_cell 默认同一时刻,这里对齐 atom 时间
    env.cells.upsert(c1)
    c2 = env.add_cell(topic="跑步", episode="叙事二",
                      atoms=[{"text": "每周跑步", "vec": _v(0.95, 0.3, 0, 0)}])
    c3 = env.add_cell(topic="瑜伽二", episode="叙事三",
                      atoms=[{"text": "馆搬到了滨海湾", "vec": _v(0.94, 0.34, 0, 0), "when": d3}])
    c3.t_start = c3.t_end = d3
    env.cells.upsert(c3)
    return env, c1, c2, c3


def test_weave_success_unit_at_first_member(db, monkeypatch):
    """织写成功:整链一个单元在链首成员位出场,covers=成员格全集,织文进 episode,topic=链 title。"""
    env, c1, c2, c3 = _yoga_scene(db)
    ch = _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    asm = assemble_units(pool, ChainStore(db), env.cells,
                         FakeLLM(["织文:先健身房后滨海湾。"]), query="瑜伽地点")
    units = asm.units

    assert asm.n_chains == 1 and asm.n_woven == 1 and len(units) == 2   # 织写单元 + 游离 c2(c3 被整链消费)
    w, plain = units
    assert w.cell.id == ch.id and w.cell.topic == ch.title
    assert w.cell.episode == "织文:先健身房后滨海湾。"
    assert w.covers == [c1.id, c3.id]                          # 成员格全集(链序)
    assert [a.atom.text for a in w.atoms] == ["馆在 MBS", "馆搬到了滨海湾"]   # 池内链成员
    assert w.cell.t_start.strftime("%m-%d") == "08-10" and w.cell.t_end.strftime("%m-%d") == "09-01"
    assert plain.cell.id == c2.id and not plain.covers


def test_weave_prompt_contract(db, monkeypatch):
    """织写 prompt 契约:五条硬规范进 system;query/title/链序 atom 清单/episode 原料进 user。"""
    env, c1, c2, c3 = _yoga_scene(db)
    _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    llm = RecorderLLM("织文。")
    assemble_units(pool, ChainStore(db), env.cells, llm, query="瑜伽地点在哪")
    assert len(llm.calls) == 1
    sys_p, user_p = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    # 五条硬规范(§5.3):不裁决/覆盖自检/episode 唯一事实源/侧重=详略/语言+人称+时间双标注
    for kw in ("never adjudicate", "Coverage first", "only fact source",
               "detail level, not selection", "Third-person", "absolute date"):
        assert kw in sys_p, kw
    # 输入四件:侧重依据 + 链 title + atom 清单(链序带日期)+ episode 原料
    assert "瑜伽地点在哪" in user_p
    assert "Chain: Caroline 的瑜伽地点" in user_p
    assert user_p.index("[2026-08-10] 馆在 MBS") < user_p.index("[2026-09-01] 馆搬到了滨海湾")
    assert "叙事一" in user_p and "叙事三" in user_p
    assert "叙事二" not in user_p                            # 游离格不进织写原料


def test_weave_failure_degrades_to_member_cells(db, monkeypatch):
    """织写调用失败 → 该链 WARNING 降级:链 atoms 回各自格桶,信息不丢,不阻塞。"""
    env, c1, c2, c3 = _yoga_scene(db)
    _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    asm = assemble_units(pool, ChainStore(db), env.cells, BoomLLM(), query="q")
    assert asm.n_woven == 0 and asm.n_plain == 3
    assert [u.cell.id for u in asm.units] == [c1.id, c2.id, c3.id]   # 格式退回整合前
    assert all(not u.covers for u in asm.units)
    assert asm.units[0].atoms[0].atom.text == "馆在 MBS"             # 链 atom 回到自家格单元


def test_weave_truncates_episodes_to_recent_15(db, monkeypatch):
    """成员格超 15:原料取最近 15 格并标注截断;atom 清单仍是整链,covers 仍是全集。"""
    env = Env(db)
    cells = []
    for i in range(17):
        day = datetime(2026, 8, 1 + i, tzinfo=timezone.utc)
        c = env.add_cell(topic=f"格{i}", episode=f"第{i}话",
                         atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0), "when": day}])
        c.t_start = c.t_end = day                    # add_cell 默认同一时刻,这里逐格错开
        env.cells.upsert(c)
        cells.append(c)
    _chain(db, env, "长链", [(c.id, f"事实{i}") for i, c in enumerate(cells)])

    llm = RecorderLLM("织文。")
    asm = assemble_units(_pool_of(env, [(c.id, f"事实{i}") for i, c in enumerate(cells)]),
                         ChainStore(db), env.cells, llm, query="q")
    assert len(llm.calls) == 1
    user_p = llm.calls[0][1]["content"]
    assert "most recent 15 of 17 segments" in user_p       # 头部标注截断
    assert "第0话" not in user_p and "第1话" not in user_p  # 最老两格的 episode 不进原料
    assert "第2话" in user_p and "第16话" in user_p         # 最近 15 格在场
    assert "[2026-08-01] 事实0" in user_p                   # atom 清单仍是整链(覆盖自检口径)
    assert asm.units[0].covers == [c.id for c in cells]    # covers 全集不截断


def test_two_chains_weave_both(db, monkeypatch):
    """池命中两条链 → 各自织写(callable 按链 title 区分响应),互不干扰。"""
    env = Env(db)
    c1 = env.add_cell(topic="甲一", episode="甲叙事一", atoms=[{"text": "甲事实一", "vec": _v(1, 0, 0, 0)}])
    c2 = env.add_cell(topic="甲二", episode="甲叙事二", atoms=[{"text": "甲事实二", "vec": _v(0.99, 0.1, 0, 0)}])
    c3 = env.add_cell(topic="乙一", episode="乙叙事一", atoms=[{"text": "乙事实一", "vec": _v(0.98, 0.2, 0, 0)}])
    c4 = env.add_cell(topic="乙二", episode="乙叙事二", atoms=[{"text": "乙事实二", "vec": _v(0.97, 0.24, 0, 0)}])
    cha = _chain(db, env, "链甲", [(c1.id, "甲事实一"), (c2.id, "甲事实二")])
    chb = _chain(db, env, "链乙", [(c3.id, "乙事实一"), (c4.id, "乙事实二")])

    def pick(p):                       # 并发下响应与链对应靠 title(与队列次序解耦)
        return "织甲" if "链甲" in p else "织乙"
    llm = FakeLLM([pick, pick])        # 两条同判别:队列够两条织写调用
    pool = _pool_of(env, [(c1.id, "甲事实一"), (c2.id, "甲事实二"),
                          (c3.id, "乙事实一"), (c4.id, "乙事实二")])
    asm = assemble_units(pool, ChainStore(db), env.cells, llm, query="q")

    assert asm.n_woven == 2 and len(asm.units) == 2
    by_id = {u.cell.id: u for u in asm.units}
    assert by_id[cha.id].cell.episode == "织甲" and by_id[cha.id].covers == [c1.id, c2.id]
    assert by_id[chb.id].cell.episode == "织乙" and by_id[chb.id].covers == [c3.id, c4.id]


# —— 残缺提示(D-C10 通用版)——

def test_boundary_from_beyond_with_chain_titles(db, monkeypatch):
    """池外 atom 挂 ≥2 节点链 → 残缺提示:数量 + 链 title;池外空 → 无提示。"""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(32)]
    # 池外两格(名次最后两名)挂进同一条 2-atom 链 → title 进提示
    _chain(db, env, "池外链标题", [(ids[30], "事实30"), (ids[31], "事实31")])

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool[:30], ChainStore(db), env.cells, None,
                         query="q", beyond=pool[30:])
    assert "2 more matching atoms exist beyond the shown materials" in asm.boundary
    assert "(chains: 池外链标题)" in asm.boundary
    assert "enumerate what is shown" in asm.boundary

    tight = assemble_units(pool[:3], ChainStore(db), env.cells, None, query="q")
    assert tight.boundary == ""                                # 无池外 → 无提示


def test_boundary_filters_covered_facts(db, monkeypatch):
    """残缺提示零噪声:已织链的池外成员(织写覆盖全链)、所在格已在材料里的池外 atom,
    都不算缺料——不进计数也不进点名,避免给下游假的"缺料"信号。"""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(33)]
    # 链甲:池内两成员(织写成功) + 池外一成员 → 池外成员事实已被织文覆盖,不算缺
    _chain(db, env, "链甲", [(ids[0], "事实0"), (ids[1], "事实1"), (ids[30], "事实30")])
    # 池外 atom ids[31] 所在格未被任何单元覆盖 → 真缺料,计数 1
    # ids[32] 的格……同样池外且未覆盖 → 计数 2(对照:过滤前是 3)
    monkeypatch.setattr("personos.online.chain_face.weave_chain", lambda *a, **k: "织文")

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool[:30], ChainStore(db), env.cells, object(),   # llm 非 None 才织写
                         query="q", beyond=pool[30:])
    assert asm.n_woven == 1                                     # 链甲织写成功
    assert "2 more matching atoms" in asm.boundary              # 事实30 已被织文覆盖,不计
    assert "链甲" not in asm.boundary                           # 已织链不进点名

    # 池外 atom 所在格已被普通单元覆盖:同格另一 atom 在池内 → 该格 episode 在材料里,不算缺
    env2 = Env(db)
    c = env2.add_cell(topic="富格", episode="叙事",
                      atoms=[{"text": "池内", "vec": _v(1, 0, 0, 0)},
                             {"text": "池外", "vec": _v(0.9, 0.1, 0, 0)}])
    pool2 = _pool_of(env2, [(c.id, "池内")])
    beyond2 = _pool_of(env2, [(c.id, "池外")])
    asm2 = assemble_units(pool2, ChainStore(db), env2.cells, None,
                          query="q", beyond=beyond2)
    assert asm2.boundary == ""                                  # 同格事实在 episode 里,不报警


def test_boundary_without_chains_has_no_title_part(db, monkeypatch):
    """池外 atom 全游离:提示只报数量,无 chains 括号。"""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(31)]
    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])

    asm = assemble_units(pool[:30], ChainStore(db), env.cells, None,
                         query="q", beyond=pool[30:])
    assert asm.boundary.startswith("1 more matching atoms exist beyond the shown materials.")
    assert "chains:" not in asm.boundary
