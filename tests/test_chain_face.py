"""Unit assembly tests (docs/atom-chain-design.md §5.2/§5.3): bucketing, dedup, weaving
with degradation, and the incomplete-materials note.

No real LLM gateway is called: assembly itself needs zero LLM calls (single-node chains and
unchained atoms); weaving uses test doubles (RecorderLLM asserts the prompt contract, a
callable FakeLLM tells the two chains apart, BoomLLM asserts the degrade path).
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
    """A test double that records every messages list, used to assert the weaving prompt
    contract (both the system and the user message are checked)."""

    def __init__(self, resp: str):
        self.resp = resp
        self.calls: list[list[dict]] = []

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        self.calls.append(messages)
        return self.resp


class CountingLLM:
    """Counts calls only: asserts that single-node chains and unchained atoms never call
    the LLM."""

    calls = 0

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        CountingLLM.calls += 1
        return "织文"


class BoomLLM:
    """chat always raises: asserts that a failed weave degrades only that one chain."""

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        raise RuntimeError("maas down")


def _pool_of(env, items: list[tuple[str, str]]):
    """Build an AtomHit pool from real stored atoms, in the given (cell_id, text) order,
    with decreasing similarity and rrf scores."""
    out = []
    for i, (cid, text) in enumerate(items):
        a = next(x for x in env.atoms.list_by_cell(cid) if x.text == text)
        out.append(AtomHit(atom=a, similarity=0.9 - i * 0.01, rrf=0.03 - i * 0.001))
    return out


def _chain(db, env, title: str, specs: list[tuple[str, str]]):
    """Build a chain from real stored atoms picked by (cell_id, text), writing through
    ChainStore directly instead of going through chain judgement. Returns the latest
    ChainInfo."""
    cs = ChainStore(db)
    members = [next(a for a in env.atoms.list_by_cell(cid) if a.text == t)
               for cid, t in specs]
    info = ChainInfo(title=title)
    cs.create_chain(info, members[0], centroid=None)
    for m in members[1:]:
        cs.append_atom(info, m, centroid=None)
    return cs.get_chain(info.id)


# -- Plain units (the general shape when the store holds no chains) --

def test_plain_units_no_chains(db):
    """With no chains in the store the materials are just the cells the pooled atoms belong
    to, deduplicated and kept in first-appearance order, with zero LLM calls."""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(33)]

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    units = asm.units

    assert asm.n_pool == 33 and asm.n_chains == 0 and asm.n_woven == 0
    assert [u.cell.id for u in units] == ids                    # first-appearance order, cells deduplicated
    assert all(isinstance(u, CellHit) and not u.covers for u in units)
    assert asm.boundary == ""                                   # nothing beyond the pool -> no note


def test_same_cell_multiple_atoms_dedup(db):
    """Several atoms of the same cell in the pool collapse into one plain unit (cells are
    deduplicated); the unit's atoms are ordered by descending similarity."""
    env = Env(db)
    c1 = env.add_cell(topic="一格多 atom", episode="叙事",
                      atoms=[{"text": "a", "vec": _v(1, 0, 0, 0)},
                             {"text": "b", "vec": _v(0.99, 0.1, 0, 0)},
                             {"text": "c", "vec": _v(0.98, 0.2, 0, 0)}])
    c2 = env.add_cell(topic="另一格", episode="叙事二", atoms=[{"text": "d", "vec": _v(0.9, 0.4, 0, 0)}])

    pool = _pool_of(env, [(c1.id, "c"), (c2.id, "d"), (c1.id, "a"), (c1.id, "b")])
    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    assert asm.n_pool == 4
    assert [u.cell.id for u in asm.units] == [c1.id, c2.id]     # c1's three atoms yield one unit
    # unit atoms sorted by descending similarity: c(0.90) > a(0.87) > b(0.86); d belongs to c2
    assert [ah.atom.text for ah in asm.units[0].atoms] == ["c", "a", "b"]


def test_orphan_cell_skipped(db):
    """An orphan atom whose cell no longer exists is skipped without crashing and without
    producing an empty shell unit."""
    env = Env(db)
    c = env.add_cell(topic="健在", episode="叙事",
                     atoms=[{"text": "好", "vec": _v(1, 0, 0, 0)}])
    env.atoms.upsert(MemoryAtom(memcell_id="cell_ghost", text="孤儿"), )
    ghost = next(a for a in env.atoms.list(limit=99) if a.text == "孤儿")
    pool = [AtomHit(atom=next(a for a in env.atoms.list_by_cell(c.id)), similarity=0.9, rrf=0.03),
            AtomHit(atom=ghost, similarity=0.8, rrf=0.02)]

    asm = assemble_units(pool, ChainStore(db), env.cells, None, query="q")
    assert [u.cell.id for u in asm.units] == [c.id]             # orphan skipped, live cell kept


# -- Chain mapping: single-node chains cost no LLM call, chains of >=2 nodes are woven --


def test_single_node_chain_is_plain_no_llm(db):
    """A single-node chain is just a plain unit for its memcell and costs no LLM call; a
    dangling chain_id is treated the same way as an unchained atom."""
    env = Env(db)
    c4 = env.add_cell(topic="独链", episode="叙事四",
                      atoms=[{"text": "单条事实", "vec": _v(0.93, 0.36, 0, 0)}])
    _chain(db, env, "单 atom 链", [(c4.id, "单条事实")])     # n_atoms=1 -> plain-unit semantics

    CountingLLM.calls = 0
    asm = assemble_units(_pool_of(env, [(c4.id, "单条事实")]),
                         ChainStore(db), env.cells, CountingLLM(), query="q")
    assert CountingLLM.calls == 0                              # a single-node chain calls no LLM
    assert [u.cell.id for u in asm.units] == [c4.id] and not asm.units[0].covers


def test_mixed_cell_chain_and_plain_coexist(db):
    """When one cell holds both a chained atom and an unchained one, the woven unit and that
    cell's plain unit coexist: the chained fact goes into the woven text while the cell's own
    episode stays complete."""
    env = Env(db)
    c1 = env.add_cell(topic="混合格", episode="叙事一",
                      atoms=[{"text": "链内事实", "vec": _v(1, 0, 0, 0)},
                             {"text": "游离事实", "vec": _v(0.97, 0.2, 0, 0)}])
    c2 = env.add_cell(topic="链友格", episode="叙事二",
                      atoms=[{"text": "链内事实二", "vec": _v(0.96, 0.28, 0, 0)}])
    ch = _chain(db, env, "一条链", [(c1.id, "链内事实"), (c2.id, "链内事实二")])   # the unchained atom stays off the chain

    asm = assemble_units(_pool_of(env, [(c1.id, "链内事实"), (c1.id, "游离事实"), (c2.id, "链内事实二")]),
                         ChainStore(db), env.cells, FakeLLM(["织文。"]), query="q")
    assert asm.n_woven == 1 and len(asm.units) == 2            # woven unit + c1's plain unit (c2 consumed by the chain)
    woven, plain = asm.units
    assert woven.cell.id == ch.id and plain.cell.id == c1.id
    assert [ah.atom.text for ah in plain.atoms] == ["游离事实"]  # the plain unit carries only the unchained atom


# -- Woven-unit semantics (citation expansion via covers; mN labels) --

def test_woven_unit_citation_expansion():
    """When a woven unit (whose covers list its member cells) is cited as mN, the citation
    expands to every member cell id, deduplicated across units."""
    cc = MemCell(id="cell_c", topic="C")
    woven = CellHit(cell=MemCell(id="chn_1", topic="链题"), score=2.0, best_sim=0.9,
                    covers=["cell_a", "cell_b"])
    plain = CellHit(cell=cc, score=1.0, best_sim=0.8)
    llm = FakeLLM(['{"answer":"织格+普通格。","cells":["m1","m2","m1"]}'])

    ans = answer_from_cells(llm, query="q", subject="", hits=[woven, plain])
    assert ans.cited_cells == ["cell_a", "cell_b", "cell_c"]   # m1 expands, and citing m1 twice does not double-count


def test_cell_block_uniform_lead():
    """Uniform rendering: a woven memcell uses the same material header format as a plain
    cell, with the chain title in the topic slot and the time range as its span."""
    c = MemCell(id="chn_1", topic="链题", episode="织文。",
                t_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                t_end=datetime(2026, 8, 30, tzinfo=timezone.utc))
    block = cell_block(CellHit(cell=c, score=1.0, best_sim=0.9, covers=["cell_a"]), "m1")
    assert "━━━ m1 ━━━" in block
    assert "[dialogue 2026-08-01 to 2026-08-30 | topic: 链题]" in block   # same format as a plain cell
    assert "[woven" not in block and "chain:" not in block                # no special header
    plain = cell_block(CellHit(cell=MemCell(id="cell_b", topic="题", episode="叙"), score=1.0,
                               best_sim=0.9), "m2")
    assert "[dialogue date unknown | topic: 题]" in plain


# -- The weaver (S4, §5.3) --

def _yoga_scene(db):
    """A three-cell scene: c1 and c3 share a chain (yoga studio location) while c2 is
    unchained (running). Returns (env, c1, c2, c3)."""
    env = Env(db)
    d1, d3 = datetime(2026, 8, 10, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)
    c1 = env.add_cell(topic="瑜伽一", episode="叙事一",
                      atoms=[{"text": "馆在 MBS", "vec": _v(1, 0, 0, 0), "when": d1}])
    c1.t_start = c1.t_end = d1                      # add_cell defaults to one instant; align it with the atom's time
    env.cells.upsert(c1)
    c2 = env.add_cell(topic="跑步", episode="叙事二",
                      atoms=[{"text": "每周跑步", "vec": _v(0.95, 0.3, 0, 0)}])
    c3 = env.add_cell(topic="瑜伽二", episode="叙事三",
                      atoms=[{"text": "馆搬到了滨海湾", "vec": _v(0.94, 0.34, 0, 0), "when": d3}])
    c3.t_start = c3.t_end = d3
    env.cells.upsert(c3)
    return env, c1, c2, c3


def test_weave_success_unit_at_first_member(db, monkeypatch):
    """On a successful weave the whole chain appears as one unit at the position of its first
    member, covers holds the full set of member cells, the woven text becomes the episode and
    the topic is the chain title."""
    env, c1, c2, c3 = _yoga_scene(db)
    ch = _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    asm = assemble_units(pool, ChainStore(db), env.cells,
                         FakeLLM(["织文:先健身房后滨海湾。"]), query="瑜伽地点")
    units = asm.units

    assert asm.n_chains == 1 and asm.n_woven == 1 and len(units) == 2   # woven unit + unchained c2 (c3 consumed by the chain)
    w, plain = units
    assert w.cell.id == ch.id and w.cell.topic == ch.title
    assert w.cell.episode == "织文:先健身房后滨海湾。"
    assert w.covers == [c1.id, c3.id]                          # all member cells, in chain order
    assert [a.atom.text for a in w.atoms] == ["馆在 MBS", "馆搬到了滨海湾"]   # chain members present in the pool
    assert w.cell.t_start.strftime("%m-%d") == "08-10" and w.cell.t_end.strftime("%m-%d") == "09-01"
    assert plain.cell.id == c2.id and not plain.covers


def test_weave_prompt_contract(db, monkeypatch):
    """The weaving prompt contract: the five hard rules go into the system message, while the
    query, chain title, chain-ordered atom list and episode source text go into the user
    message."""
    env, c1, c2, c3 = _yoga_scene(db)
    _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    llm = RecorderLLM("织文。")
    assemble_units(pool, ChainStore(db), env.cells, llm, query="瑜伽地点在哪")
    assert len(llm.calls) == 1
    sys_p, user_p = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    # the five hard rules (§5.3): never adjudicate; self-check coverage; episode is the only
    # fact source; emphasis means detail level, not selection; language, person and dual date marking
    for kw in ("never adjudicate", "Coverage first", "only fact source",
               "detail level, not selection", "Third-person", "absolute date"):
        assert kw in sys_p, kw
    # four inputs: the emphasis cue, the chain title, the dated atom list in chain order, and the episode source text
    assert "瑜伽地点在哪" in user_p
    assert "Chain: Caroline 的瑜伽地点" in user_p
    assert user_p.index("[2026-08-10] 馆在 MBS") < user_p.index("[2026-09-01] 馆搬到了滨海湾")
    assert "叙事一" in user_p and "叙事三" in user_p
    assert "叙事二" not in user_p                            # an unchained cell is not fed to the weaver


def test_weave_failure_degrades_to_member_cells(db, monkeypatch):
    """A failed weave call degrades that chain with a WARNING: its atoms fall back into their
    own cell buckets, so no information is lost and nothing blocks."""
    env, c1, c2, c3 = _yoga_scene(db)
    _chain(db, env, "Caroline 的瑜伽地点", [(c1.id, "馆在 MBS"), (c3.id, "馆搬到了滨海湾")])

    pool = _pool_of(env, [(c1.id, "馆在 MBS"), (c2.id, "每周跑步"), (c3.id, "馆搬到了滨海湾")])
    asm = assemble_units(pool, ChainStore(db), env.cells, BoomLLM(), query="q")
    assert asm.n_woven == 0 and asm.n_plain == 3
    assert [u.cell.id for u in asm.units] == [c1.id, c2.id, c3.id]   # shape falls back to the pre-assembly form
    assert all(not u.covers for u in asm.units)
    assert asm.units[0].atoms[0].atom.text == "馆在 MBS"             # the chained atom returns to its own cell's unit


def test_weave_truncates_episodes_to_recent_15(db, monkeypatch):
    """With more than 15 member cells the source text keeps only the 15 most recent ones and
    says so; the atom list still covers the whole chain and covers is still the full set."""
    env = Env(db)
    cells = []
    for i in range(17):
        day = datetime(2026, 8, 1 + i, tzinfo=timezone.utc)
        c = env.add_cell(topic=f"格{i}", episode=f"第{i}话",
                         atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0), "when": day}])
        c.t_start = c.t_end = day                    # add_cell defaults to one instant; stagger the cells day by day
        env.cells.upsert(c)
        cells.append(c)
    _chain(db, env, "长链", [(c.id, f"事实{i}") for i, c in enumerate(cells)])

    llm = RecorderLLM("织文。")
    asm = assemble_units(_pool_of(env, [(c.id, f"事实{i}") for i, c in enumerate(cells)]),
                         ChainStore(db), env.cells, llm, query="q")
    assert len(llm.calls) == 1
    user_p = llm.calls[0][1]["content"]
    assert "most recent 15 of 17 segments" in user_p       # the header states the truncation
    assert "第0话" not in user_p and "第1话" not in user_p  # the two oldest episodes are left out
    assert "第2话" in user_p and "第16话" in user_p         # the 15 most recent cells are present
    assert "[2026-08-01] 事实0" in user_p                   # the atom list still spans the whole chain, so coverage self-check works
    assert asm.units[0].covers == [c.id for c in cells]    # covers is the full set, never truncated


def test_two_chains_weave_both(db, monkeypatch):
    """When the pool hits two chains each is woven separately, without interfering with the
    other; the callable double picks its response by chain title."""
    env = Env(db)
    c1 = env.add_cell(topic="甲一", episode="甲叙事一", atoms=[{"text": "甲事实一", "vec": _v(1, 0, 0, 0)}])
    c2 = env.add_cell(topic="甲二", episode="甲叙事二", atoms=[{"text": "甲事实二", "vec": _v(0.99, 0.1, 0, 0)}])
    c3 = env.add_cell(topic="乙一", episode="乙叙事一", atoms=[{"text": "乙事实一", "vec": _v(0.98, 0.2, 0, 0)}])
    c4 = env.add_cell(topic="乙二", episode="乙叙事二", atoms=[{"text": "乙事实二", "vec": _v(0.97, 0.24, 0, 0)}])
    cha = _chain(db, env, "链甲", [(c1.id, "甲事实一"), (c2.id, "甲事实二")])
    chb = _chain(db, env, "链乙", [(c3.id, "乙事实一"), (c4.id, "乙事实二")])

    def pick(p):                       # under concurrency the response is matched to the chain by title, not by queue order
        return "织甲" if "链甲" in p else "织乙"
    llm = FakeLLM([pick, pick])        # same discriminator twice: the queue holds enough entries for two weave calls
    pool = _pool_of(env, [(c1.id, "甲事实一"), (c2.id, "甲事实二"),
                          (c3.id, "乙事实一"), (c4.id, "乙事实二")])
    asm = assemble_units(pool, ChainStore(db), env.cells, llm, query="q")

    assert asm.n_woven == 2 and len(asm.units) == 2
    by_id = {u.cell.id: u for u in asm.units}
    assert by_id[cha.id].cell.episode == "织甲" and by_id[cha.id].covers == [c1.id, c2.id]
    assert by_id[chb.id].cell.episode == "织乙" and by_id[chb.id].covers == [c3.id, c4.id]


# -- The incomplete-materials note (general form, D-C10) --

def test_boundary_from_beyond_with_chain_titles(db, monkeypatch):
    """When atoms beyond the pool belong to a chain of at least two nodes, the note reports
    both a count and the chain titles; when nothing lies beyond the pool there is no note."""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(32)]
    # the two lowest-ranked cells, both beyond the pool, share one 2-atom chain -> its title shows up in the note
    _chain(db, env, "池外链标题", [(ids[30], "事实30"), (ids[31], "事实31")])

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool[:30], ChainStore(db), env.cells, None,
                         query="q", beyond=pool[30:])
    assert "2 more matching atoms exist beyond the shown materials" in asm.boundary
    assert "(chains: 池外链标题)" in asm.boundary
    assert "enumerate what is shown" in asm.boundary

    tight = assemble_units(pool[:3], ChainStore(db), env.cells, None, query="q")
    assert tight.boundary == ""                                # nothing beyond the pool -> no note


def test_boundary_filters_covered_facts(db, monkeypatch):
    """The incomplete-materials note must carry no noise: a beyond-pool member of an already
    woven chain (the weave covers the whole chain), and a beyond-pool atom whose cell is
    already in the materials, are both not missing. Neither is counted nor named, so the
    downstream reader never gets a false "materials are missing" signal."""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(33)]
    # first chain: two members inside the pool (woven successfully) plus one beyond it -> the
    # beyond-pool member's fact is already covered by the woven text, so it is not missing
    _chain(db, env, "链甲", [(ids[0], "事实0"), (ids[1], "事实1"), (ids[30], "事实30")])
    # the cell of beyond-pool atom ids[31] is covered by no unit -> genuinely missing, count 1
    # ids[32]'s cell is likewise beyond the pool and uncovered -> count 2 (without the filter it would be 3)
    monkeypatch.setattr("personos.online.chain_face.weave_chain", lambda *a, **k: "织文")

    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])
    asm = assemble_units(pool[:30], ChainStore(db), env.cells, object(),   # weaving only happens when llm is not None
                         query="q", beyond=pool[30:])
    assert asm.n_woven == 1                                     # the first chain was woven
    assert "2 more matching atoms" in asm.boundary              # 事实30 is already covered by the woven text, so it is not counted
    assert "链甲" not in asm.boundary                           # an already woven chain is never named

    # the beyond-pool atom's cell is already covered by a plain unit, because a sibling atom of
    # the same cell made it into the pool -> that cell's episode is in the materials, nothing is missing
    env2 = Env(db)
    c = env2.add_cell(topic="富格", episode="叙事",
                      atoms=[{"text": "池内", "vec": _v(1, 0, 0, 0)},
                             {"text": "池外", "vec": _v(0.9, 0.1, 0, 0)}])
    pool2 = _pool_of(env2, [(c.id, "池内")])
    beyond2 = _pool_of(env2, [(c.id, "池外")])
    asm2 = assemble_units(pool2, ChainStore(db), env2.cells, None,
                          query="q", beyond=beyond2)
    assert asm2.boundary == ""                                  # the sibling fact lives in the episode, so no warning


def test_boundary_without_chains_has_no_title_part(db, monkeypatch):
    """When every beyond-pool atom is unchained the note reports only a count, with no
    parenthesised chains part."""
    env = Env(db)
    ids = [env.add_cell(topic=f"题{i}", episode=f"叙事{i}",
                        atoms=[{"text": f"事实{i}", "vec": _v(1, 0, 0, 0)}]).id
           for i in range(31)]
    pool = _pool_of(env, [(cid, f"事实{i}") for i, cid in enumerate(ids)])

    asm = assemble_units(pool[:30], ChainStore(db), env.cells, None,
                         query="q", beyond=pool[30:])
    assert asm.boundary.startswith("1 more matching atoms exist beyond the shown materials.")
    assert "chains:" not in asm.boundary
