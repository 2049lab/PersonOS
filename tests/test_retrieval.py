"""Fast-path unit tests: R0 parsing of the five outputs and its degradation / R1 two-route atom
retrieval with RRF and pool selection / R5 answer citations and degradation.

No real model provider is contacted: TableEmbedder makes the text-to-vector mapping fully
controlled so similarities can be worked out by hand, and FakeLLM returns queued JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from personos.models import MemCell, MemoryAtom
from personos.online.retrieval import (
    AtomHit,
    CellHit,
    QueryRewrite,
    answer_from_cells,
    cell_block,
    rewrite_query,
    search_atoms,
)
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore

from .fakes import FakeLLM

_T = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


def _v(*x: float) -> np.ndarray:
    return np.array(x, dtype=np.float32)


class TableEmbedder:
    """Text to a preset vector; text not in the table gets the zero vector (it never calls a
    remote service)."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = {k: _v(*v) for k, v in table.items()}
        self.dim = len(next(iter(self.table.values())))

    def embed(self, texts: list[str]) -> np.ndarray:
        zero = np.zeros(self.dim, dtype=np.float32)
        return np.vstack([self.table.get(t, zero) for t in texts])


class Env:
    """An isolated database plus cell/atom stores, for hand-building cells and atoms with
    vectors as needed."""

    def __init__(self, db):
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)

    def add_cell(self, *, topic="t", episode="e", domains=(), atoms=(), topic_vec=None):
        """atoms is [{text, vec, domains?, when?}] and gets persisted into a cell. topic_vec
        optionally stores a topic vector (for the topic route)."""
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
        """Assemble a CellHit by hand, for the R5/R3 material tests that bypass retrieval."""
        return CellHit(cell=cell, score=0.0, best_sim=0.0,
                       atoms=[AtomHit(atom=a, similarity=0.5) for a in atoms])


def _rw(resolved="q", expansions=(), domains=()):
    return QueryRewrite(original="q", resolved=resolved,
                        expansions=list(expansions), domains=list(domains))


# -- R1: two-route atom retrieval + RRF + pool selection --

def test_atom_pool_ranked_by_maxsim(db):
    """The atom pool is ranked by the maximum cosine against the query faces (expansion terms
    included); the unit in the pool is the atom, no longer the cell."""
    env = Env(db)
    c1 = env.add_cell(topic="画展", atoms=[
        {"text": "弱相关", "vec": _v(0.6, 0.8, 0, 0)},
        {"text": "强相关", "vec": _v(1, 0, 0, 0)},
    ])
    env.add_cell(topic="跑步机", atoms=[{"text": "中等", "vec": _v(0.8, 0.6, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    # Descending similarity.
    assert [ah.atom.text for ah in pool.atoms] == ["强相关", "中等", "弱相关"]
    assert all(ah.atom.memcell_id == c1.id or ah.atom.text == "中等" for ah in pool.atoms)
    assert abs(pool.atoms[0].similarity - 1.0) < 1e-6
    assert pool.atoms[0].rrf > pool.atoms[1].rrf > pool.atoms[2].rrf
    assert pool.beyond == []                                  # all 3 atoms fit in the pool, nothing beyond


def test_expansion_face_takes_max(db):
    """Each query face on the associative route is embedded separately: an atom's score is the
    maximum cosine across all the faces, expansion terms included."""
    env = Env(db)
    env.add_cell(topic="画展", atoms=[{"text": "命中扩展面", "vec": _v(0, 1, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0), "筹备": _v(0, 1, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw(expansions=["筹备"]))
    assert pool.atoms and pool.atoms[0].atom.text == "命中扩展面"
    assert abs(pool.atoms[0].similarity - 1.0) < 1e-6         # reached via the expansion face


def test_domain_path_and_rrf_dual_hit_wins(db):
    """An atom both routes agree on floats to the top: b, which is 2nd on the associative route
    and 1st on the domain route, beats c on RRF score even though c is 1st on the associative
    route alone."""
    env = Env(db)
    env.add_cell(topic="B", domains=["D05"],
                 atoms=[{"text": "b", "vec": _v(0.9, 0.44, 0, 0), "domains": ["D05"]}])
    # Strongest semantically but carries no domain.
    env.add_cell(topic="C", atoms=[{"text": "c", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="A", domains=["D05"],
                 atoms=[{"text": "a", "vec": _v(0.5, 0.5, 0.5, 0.5), "domains": ["D05"]}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw(domains=["D05"]))
    # b (both routes) > a (both routes, lower score) > c (top of one route only).
    assert [ah.atom.text for ah in pool.atoms] == ["b", "a", "c"]
    assert pool.atoms[0].rrf > pool.atoms[2].rrf                   # two-route consensus beats a single-route leader
    # The single-route leader only lags on rrf; it is still the strongest semantically.
    assert pool.atoms[2].similarity > pool.atoms[0].similarity


def test_no_domains_degrades_to_assoc_order(db):
    """No domain could be determined, so the domain route is empty and RRF degrades to a single
    route: the pool order is just the associative similarity order."""
    env = Env(db)
    env.add_cell(topic="1", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    env.add_cell(topic="2", atoms=[{"text": "y", "vec": _v(0.5, 0.5, 0.5, 0.5)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert [ah.atom.text for ah in pool.atoms] == ["x", "y"]


def test_per_cell_cap_blocks_rich_cell(db):
    """At most 10 atoms from one cell enter the pool, which stops a rich cell flooding it and
    squeezing everyone else out. Those squeezed out do not go into beyond, because that cell is
    already present in the materials."""
    env = Env(db)
    env.add_cell(topic="富格", atoms=[{"text": f"富{i}", "vec": _v(1 - i * 0.01, 0.1, 0, 0)}
                                      for i in range(12)])
    env.add_cell(topic="弱格", atoms=[{"text": "弱", "vec": _v(0.3, 0.95, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    rich = [ah for ah in pool.atoms if ah.atom.text.startswith("富")]
    assert len(rich) == 10                                   # only 10 from the rich cell get in
    # The weak cell is not squeezed out.
    assert [ah.atom.text for ah in pool.atoms[-1:]] == ["弱"]
    assert pool.beyond == []                                 # the pool is not full, so exclusions are not missing material


def test_pool_cut_and_beyond(db):
    """Once the pool is full (30 by default) the remaining ranks go into beyond — the raw
    material for naming what the chain view is missing."""
    env = Env(db)
    for i in range(33):
        env.add_cell(topic=f"题{i}", atoms=[{"text": f"事实{i}", "vec": _v(1 - i * 0.001, 0.04, 0, 0)}])
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})

    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert len(pool.atoms) == 30
    assert [ah.atom.text for ah in pool.beyond] == ["事实30", "事实31", "事实32"]
    # Everything beyond ranks below the tail of the pool.
    assert all(ah.rrf < pool.atoms[-1].rrf for ah in pool.beyond)


def test_orphan_atom_and_empty_pool(db):
    """An orphan atom with no memcell_id does not enter the pool; an empty database returns an
    empty pool."""
    env = Env(db)
    env.atoms.upsert(MemoryAtom(memcell_id="", text="孤儿"), embedding=_v(1, 0, 0, 0))
    emb = TableEmbedder({"q": _v(1, 0, 0, 0)})
    pool = search_atoms(emb, env.atoms, rewrite=_rw())
    assert pool.atoms == [] and pool.beyond == []


# -- R0: query preprocessing --

def test_rewrite_parses_five_outputs(db):
    llm = FakeLLM(['{"resolved":"Caroline 的画展展期定在哪天","subject":"Caroline",'
                   '"expansions":["画展","筹备","展期"],"time_start":"2026-08-11",'
                   '"time_end":"2026-08-17","domains":["D13","D99"]}'])
    rw = rewrite_query(llm, raw_query="她的画展什么时候", now_dt=_T)
    assert rw.resolved == "Caroline 的画展展期定在哪天"
    assert rw.subject == "Caroline"
    assert rw.expansions == ["画展", "筹备", "展期"]
    assert (rw.time_start, rw.time_end) == ("2026-08-11", "2026-08-17")
    # D99 is not in the vocabulary, so it is dropped — this is what stops the LLM inventing domains.
    assert rw.domains == ["D13"]


def test_rewrite_normalizes_nulls(db):
    llm = FakeLLM(['{"resolved":"q","subject":"null","expansions":[],'
                   '"time_start":"null","time_end":null,"domains":[]}'])
    rw = rewrite_query(llm, raw_query="q", now_dt=_T)
    assert rw.subject == "" and rw.time_start == "" and rw.time_end == ""


def test_rewrite_parse_failure_degrades_to_original(db):
    llm = FakeLLM(["模型跑偏,不是 JSON"])
    rw = rewrite_query(llm, raw_query="原始问题", now_dt=_T)
    assert rw.resolved == "原始问题" and rw.domains == [] and rw.expansions == []
    # Traceability: even when degrading, keep the model's raw output.
    assert "模型跑偏" in rw.raw


# -- R5: answering --

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
    # m9 is an unknown handle and gets dropped.
    assert ans.cited_cells == [h1.cell.id, h2.cell.id]


def test_answer_parse_failure_outputs_raw(db):
    """All three attempts are non-JSON, so the last raw output is passed through verbatim.
    Measured in bench run conv26-v3: the evaluation model intermittently returns non-JSON, and
    R5 used to have no retry at all."""
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM(["坏一", "坏二", "坏三"])
    ans = answer_from_cells(llm, query="q", subject="", hits=[h])
    # What gets passed through is the last raw output.
    assert ans.answer == "坏三" and ans.cited_cells == []


def test_answer_infra_error_yields_empty_not_exception_text(db):
    """Infrastructure errors such as rate limiting: the exception text must never become the
    answer. Measured in H1, 'Error code: 429 ...' once leaked out as the R5 answer."""
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])

    class RateLimitedLLM:
        def chat(self, *a, **k):
            raise RuntimeError("Error code: 429 - {'type': 'rate_limit_error'}")

    ans = answer_from_cells(RateLimitedLLM(), query="q", subject="", hits=[h])
    # An empty answer lets the layer above explain the situation objectively via _no_answer_note.
    assert ans.answer == ""


def test_answer_retry_recovers_from_bad_json(db):
    """The first output is non-JSON, the retry returns valid JSON, and the answer comes through
    normally — the targeted fix for 7 failures measured in bench."""
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
    assert ans.answer == "" and llm.calls == 0                 # with no materials, the LLM is not called


def test_answer_prompt_carries_now_anchor(db):
    env = Env(db)
    h = _cell_with_atoms(env, topic="t", episode="e", atom_specs=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM(['{"answer":"答","cells":["m1"]}'])
    t0 = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
    answer_from_cells(llm, query="多久了", subject="user", hits=[h], now_dt=t0)
    assert "Current time: 2026-08-28T10:00:00+00:00" in llm.last_user_prompt
    # Not passed -> the section is absent.
    answer_from_cells(llm, query="多久了", subject="user", hits=[h])
    assert "Current time:" not in llm.last_user_prompt


def test_cell_block_renders_material(db):
    """Material format: a "== cN/mN ==" separator line, a neutral header with the dialogue time
    and topic, then the episode as the main material; atoms do not enter the materials.

    cell_block only renders — the handle is supplied by the caller, so the fast path (mN) and
    the deep track (cN) share this one renderer."""
    env = Env(db)
    c = env.add_cell(topic="画展筹备", episode="Caroline 在筹备画展,展期 2026-09。",
                     atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}])
    h = env.hit(c, env.atoms.list_by_cell(c.id))
    block = cell_block(h, "c1")
    lines = block.splitlines()
    assert lines[0] == "━━━ c1 ━━━"                             # the hard separator line, handle included
    assert lines[1] == "[dialogue 2026-08-25 | topic: 画展筹备]"
    assert "Caroline 在筹备画展" in block                         # the episode is the main material
    assert "展期定在 2026-09" not in block                        # atom text does not enter the answering materials


# -- R5: material rendering order (the P1-B switch) --

def _hit_at(topic: str, episode: str, t):
    """A CellHit that is never persisted (the ordering tests only care about t_start, so no
    store is needed)."""
    c = MemCell(topic=topic, episode=episode, domains=[], t_start=t, t_end=t)
    return CellHit(cell=c, score=0.0, best_sim=0.0, atoms=[])


def _first_block_episode(llm):
    """Pull the episode of the first material block out of the user prompt R5 received (the
    second line after the separator)."""
    return llm.last_user_prompt.split("━━━ m1 ━━━\n")[1].splitlines()[1]


def test_r5_order_env_controls_material_order(monkeypatch):
    """PERSONOS_R5_ORDER: relevance keeps the order passed in (the rerank order), time_asc
    sorts oldest first, time_desc newest first, and the default is relevance."""
    early = _hit_at("早", "一月的事", datetime(2026, 1, 1, tzinfo=timezone.utc))
    late = _hit_at("晚", "六月的事", datetime(2026, 6, 1, tzinfo=timezone.utc))
    llm = FakeLLM(['{"answer":"a","cells":["m1"]}'])

    monkeypatch.setenv("PERSONOS_R5_ORDER", "relevance")
    answer_from_cells(llm, query="q", subject="", hits=[late, early])   # rerank order: the late one first
    assert _first_block_episode(llm) == "六月的事"

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_asc")
    answer_from_cells(llm, query="q", subject="", hits=[late, early])
    assert _first_block_episode(llm) == "一月的事"                       # ascending: the early one first

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_desc")
    # The input order is reversed here, which proves sorting really happens.
    answer_from_cells(llm, query="q", subject="", hits=[early, late])
    assert _first_block_episode(llm) == "六月的事"

    monkeypatch.delenv("PERSONOS_R5_ORDER", raising=False)
    answer_from_cells(llm, query="q", subject="", hits=[late, early])
    assert _first_block_episode(llm) == "六月的事"                       # defaults to relevance (the baseline)


def test_r5_order_none_tstart_floor_and_stable_ties(monkeypatch):
    """A missing t_start counts as the earliest and sinks to the bottom; equal timestamps keep
    the order they were passed in (a stable sort), so the rerank order is still the reference
    within a tie."""
    undated = _hit_at("无期", "无时间的事", None)
    a = _hit_at("同刻A", "A 的事", _T)
    b = _hit_at("同刻B", "B 的事", _T)
    llm = FakeLLM(['{"answer":"a","cells":["m1"]}'])

    monkeypatch.setenv("PERSONOS_R5_ORDER", "time_asc")
    answer_from_cells(llm, query="q", subject="", hits=[b, undated, a])
    body = llm.last_user_prompt

    def _ep(handle):
        return body.split(f"━━━ {handle} ━━━\n")[1].splitlines()[1]

    assert _ep("m1") == "无时间的事"      # no timestamp -> the sort floor, so earliest
    # Equal timestamps sort stably: the order passed in is preserved.
    assert _ep("m2") == "B 的事" and _ep("m3") == "A 的事"


def test_answer_prompt_carries_p1_rules():
    """Guard over the P1 clauses: "trust the most recent" for consuming conflicts, and
    "count first, then verify" for enumeration, must not be quietly dropped by a later rewrite."""
    from personos.online.retrieval import _ANSWER_SYS, CONFLICT_RULE
    assert "the MOST RECENT statement is the current state" in CONFLICT_RULE
    assert "count the distinct items the materials actually contain" in _ANSWER_SYS
    assert "verify your list has exactly that many" in _ANSWER_SYS
