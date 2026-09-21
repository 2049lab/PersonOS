"""Deep-track unit tests: the renderers / the handle registry / the six tools (retrieval runs a
coarse pass then a rerank, and search_evidence is the fallback that searches raw utterances
directly) / the MaasChatModel adapter / the run_deep agent loop.

No real model provider is contacted: TableEmbedder controls the vectors, and FakeLLM or a stub
LLM returns queued JSON tool-call blocks to drive the agent loop.
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
    """Text to a preset vector; text not in the table gets the zero vector (it never calls a
    remote service)."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = {k: _v(*v) for k, v in table.items()}
        self.dim = len(next(iter(self.table.values())))

    def embed(self, texts: list[str]) -> np.ndarray:
        zero = np.zeros(self.dim, dtype=np.float32)
        return np.vstack([self.table.get(t, zero) for t in texts])


class ZeroEmbedder:
    """A zero-vector embedder, for cases that do not care about semantics."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class Env:
    """An isolated database plus the three stores; add_cell supports atoms with vectors and
    evidence with raw utterances."""

    def __init__(self, db):
        self.cells = CellStore(db)
        self.atoms = AtomStore(db)
        self.ev = EvidenceStore(db)

    def add_cell(self, *, topic="t", episode="e", t_start=_T, domains=(),
                 atoms=(), evidence=(), topic_vec=None) -> MemCell:
        """atoms is [{text, vec, domains?, holder?, when?}]; evidence is [(holder, utterance)]."""
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


# -- The handle registry --

def test_registry_assigns_stable_handles():
    reg = HandleRegistry()
    assert reg.ensure("cell_a") == "c1"
    assert reg.ensure("cell_b") == "c2"
    assert reg.ensure("cell_a") == "c1"          # ensuring again does not change the handle
    assert reg.real("c2") == "cell_b"
    assert reg.real(" c1 ") == "cell_a"          # surrounding whitespace is tolerated
    assert reg.real("c9") is None and len(reg) == 2


# -- The renderers --

def test_cell_full_and_row_hide_atom_text(db):
    env = Env(db)
    c = env.add_cell(topic="画展筹备", episode="Caroline 在筹备画展,展期 2026-09。",
                     atoms=[{"text": "展期定在 2026-09", "vec": _v(1, 0, 0, 0)}],
                     evidence=[("Caroline", "展期就定九月")])
    full = cell_full(c, "c3", n_atoms=5, n_lines=2)
    assert full.splitlines()[0] == "━━━ c3 ━━━"
    assert "this segment has 5 extracted index atoms and 2 raw utterances" in full
    assert "get_cell_evidence" in full                          # points the way to verifying against raw utterances
    # The catalog row carries a second-resolution timestamp.
    assert cell_row("c3", c) == "c3 | 2026-08-20 09:00:00 | 画展筹备"


def test_evidence_page_paginates_and_names_speaker(db):
    recs = [EvidenceRecord(holder="Caroline" if i % 2 else "user",
                           content_inline=f"第{i}句", captured_at=_T)
            for i in range(35)]
    body, total = evidence_page(recs, page=1)
    assert total == 2                                            # 35 utterances -> 2 pages (30 per page)
    assert body.count("\n") == 29                                # the first page holds a full 30 lines
    body2, _ = evidence_page(recs, page=2)
    assert "第30句" in body2 and "第34句" in body2
    assert "[2026-08-20 09:00:00] Caroline: " in body            # the speaker is the holder's real name
    empty, total_e = evidence_page([], page=1)
    assert "no raw utterances kept" in empty and total_e == 1


# -- search_atoms: filter, MaxSim coarse pass, then rerank --

def test_search_atoms_window_filter_and_payload_shape(db):
    env = Env(db)
    # Atoms inside the window are given an explicit when: if the anchor fell back to
    # recorded_at = now(), it would drift out of the August window as the calendar rolls over
    # into the next month (the calendar-sensitivity fix).
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
    assert "旧叙事" not in out                                  # the date window excludes the January cell
    assert "画展叙事。" in out and "跑步叙事。" in out          # weak hits still enter the materials, within the limit
    assert "展期 2026-09" not in out and "跑三公里" not in out  # atom text never appears
    assert "━━━ c1 ━━━" in out and "━━━ c2 ━━━" in out          # matched cells have been assigned handles


def test_search_atoms_rerank_reorders_candidates(db):
    env = Env(db)
    strong = env.add_cell(topic="画展", episode="画展叙事。",
                          atoms=[{"text": "展期", "vec": _v(1, 0, 0, 0)}])
    weak = env.add_cell(topic="跑步", episode="跑步叙事。",
                        atoms=[{"text": "配速", "vec": _v(0.6, 0.8, 0, 0)}])
    d = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}))
    noop_out = tool_search_atoms(d, query="展期")
    assert noop_out.index("画展叙事。") < noop_out.index("跑步叙事。")   # Noop: MaxSim order

    class FlipReranker:   # forcibly reverses the order, proving the rerank really owns the final order
        def rerank(self, query, documents, *, instruction=""):
            return list(reversed([0.9 / (i + 1) for i in range(len(documents))]))

    d2 = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}), reranker=FlipReranker())
    flip_out = tool_search_atoms(d2, query="展期")
    assert flip_out.index("跑步叙事。") < flip_out.index("画展叙事。")   # the rerank order took effect


def test_search_atoms_empty_gives_actionable_advice(db):
    env = Env(db)
    env.add_cell(topic="画展", episode="叙事。",
                 atoms=[{"text": "展期", "vec": _v(1, 0, 0, 0), "domains": ["D13"]}])
    d = env.deps(TableEmbedder({"展期": _v(1, 0, 0, 0)}))
    out = tool_search_atoms(d, query="展期", domains=["D99"])    # the domain filter empties the pool
    assert "No hits" in out and "find_cells" in out


def test_search_atoms_limit_clamped_to_8(db):
    env = Env(db)
    for i in range(12):
        env.add_cell(topic=f"t{i}", atoms=[{"text": f"x{i}", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(TableEmbedder({"q": _v(1, 0, 0, 0)}))
    out = tool_search_atoms(d, query="q", limit=99)
    assert "search_atoms hit 8 unit(s)" in out                  # hard cap of 8


# -- search_atoms chain expansion (S5, §6) --

def _mk_chain(db, env, title, specs):
    """Build a chain from the real atoms in the database, looked up by (cell_id, text), going
    straight to ChainStore rather than through chain assignment."""
    cs = ChainStore(db)
    members = [next(a for a in env.atoms.list_by_cell(cid) if a.text == t)
               for cid, t in specs]
    info = ChainInfo(title=title)
    cs.create_chain(info, members[0], centroid=None)
    for m in members[1:]:
        cs.append_atom(info, m, centroid=None)
    return cs.get_chain(info.id)


class _StubLLM:
    """A test double for the weaver: always returns the same woven text (the chat signature
    matches the ChatLLM protocol)."""
    def __init__(self, text="织文:馆先在健身房,后搬到MBS。"):
        self.text = text
    def chat(self, messages, temperature=0.3, max_tokens=2048):
        return self.text


def test_search_atoms_weaves_chain_members(db, monkeypatch):
    """Hitting a chain with 2 or more nodes weaves a memcell-prime by the same mechanism as the
    fast path: the woven text replaces the scattered cells, the block header lists the member
    cell handles, and every member cell is registered so it can be drilled into. Atom text does
    not appear, and unrelated cells are not dragged along."""
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
    assert "织文:馆先在健身房" in out                            # the woven unit (a merged narrative)
    assert "瑜伽叙事一。" not in out and "瑜伽叙事二。" not in out  # the scattered cells were replaced by the woven text
    assert "woven from fact-chains" in out
    h1, h3 = d.reg.real("c1"), d.reg.real("c2")
    assert {h1, h3} == {c1.id, c3.id}                           # both member cells are registered, so open works
    assert f"━━━ c1, c2 ━━━" in out                             # the woven block header lists several handles side by side
    assert "跑步叙事。" in out                                  # ordinary units coexist with it
    assert "(pulled via chain" not in out                       # the old scattered-cell expansion has been removed


def test_search_atoms_weave_covers_whole_long_chain(db, monkeypatch):
    """A long chain is not truncated: hitting an 8-member chain produces a woven unit covering
    all 8 cells (the old implementation capped at 5 and truncated silently)."""
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
    handles = [d.reg.ensure(c.id) for c in env.cells.iter_all()]  # all 8 cells should already be registered
    assert len(handles) == 8
    assert "c1, c2, c3, c4, c5, c6, c7, c8" in out               # the block header lists every member cell


def test_search_atoms_chain_degrades_without_llm(db, monkeypatch):
    """With no llm (in tests, or when degrading), chain members fall back to ordinary cell
    units — the same mechanism as the fast path's degradation route."""
    env = Env(db)
    c1 = env.add_cell(topic="瑜伽一", episode="瑜伽叙事一。",
                      atoms=[{"text": "馆在健身房", "vec": _v(1, 0, 0, 0)}])
    c3 = env.add_cell(topic="瑜伽二", episode="瑜伽叙事二。",
                      atoms=[{"text": "馆搬到了MBS", "vec": _v(0.5, 0.86, 0, 0)}])
    _mk_chain(db, env, "用户瑜伽地点", [(c1.id, "馆在健身房"), (c3.id, "馆搬到了MBS")])
    d = env.deps(TableEmbedder({"瑜伽": _v(1, 0, 0, 0)}))         # llm=None
    out = tool_search_atoms(d, query="瑜伽", limit=2)
    assert "瑜伽叙事一。" in out and "瑜伽叙事二。" in out        # member cells appear as ordinary units
    assert "woven" not in out


# -- find_cells: time window and domain filters + topic similarity + pagination --

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
    # Newest first, and anything outside the domain is dropped.
    assert [d.reg.real(ln.split(" |")[0]) for ln in rows] == [aug.id, jul.id]
    assert "跑步" not in out


def test_find_cells_window_inclusive_of_end_day(db):
    env = Env(db)
    in_c = env.add_cell(topic="画展", t_start=datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc))
    env.add_cell(topic="旅行", t_start=datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc))
    edge = env.add_cell(topic="跑步", t_start=datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    out = tool_find_cells(d, start_date="2026-08-01", end_date="2026-08-31")
    assert "旅行" not in out                                    # outside the window, so dropped
    ids = {d.reg.real(ln.split(" |")[0]) for ln in out.splitlines() if ln.startswith("c")}
    assert ids == {in_c.id, edge.id}                            # the boundary days are included


def test_find_cells_query_ranks_by_topic_similarity(db):
    env = Env(db)
    env.add_cell(topic="跑步计划", topic_vec=_v(0.2, 0.98))
    near = env.add_cell(topic="画展筹备", topic_vec=_v(0.98, 0.2))
    d = env.deps(TableEmbedder({"画展": _v(1, 0)}))
    out = tool_find_cells(d, query="画展")
    assert "topic similarity descending" in out
    first = out.splitlines()[1].split(" |")[0]
    assert d.reg.real(first) == near.id                         # the closer topic ranks first


def test_find_cells_paging_tail(db):
    env = Env(db)
    for i in range(13):
        env.add_cell(topic=f"t{i:02d}", t_start=datetime(2026, 8, i + 1, tzinfo=timezone.utc))
    d = env.deps(ZeroEmbedder())
    p1 = tool_find_cells(d, page=1)
    assert "find_cells page 1/2 (13 cell(s) after filtering" in p1 and "page=2" in p1
    assert "t12" in p1 and "t00" not in p1                   # descending: newest first
    p2 = tool_find_cells(d, page=2)
    assert "(last page)" in p2 and "t00" in p2 and "t12" not in p2


# -- open_cell / get_cell_evidence --

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


# -- search_evidence: keyword search straight over raw utterances (the fallback path, which
# bypasses the index) --

def test_search_evidence_finds_utterance_atoms_missed(db):
    """The fallback scenario: a fact that exists only in the raw utterance (zero atoms were
    extracted) can still be found, and the answer names the cell it belongs to."""
    env = Env(db)
    c = env.add_cell(topic="厨房琐事", episode="聊了些家里的事。",
                     atoms=(),                                    # extraction missed it, so the index is empty
                     evidence=[("Melanie", "I broke my favourite bowl last night"),
                               ("user", "没事,再买一个就好")])
    d = env.deps(ZeroEmbedder())
    out = tool_search_evidence(d, keywords=["bowl", "broke"])
    assert "search_evidence hit 1 utterance(s)" in out
    assert "Melanie: I broke my favourite bowl last night" in out
    assert "(cell c1)" in out


def test_search_evidence_requires_all_keywords_same_sentence(db):
    """Keywords are ANDed within one sentence: two sentences each holding one word is not a
    hit."""
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
    assert args.limit == 15                                       # a bad value falls back to the default


# -- remember: additive only + quote back-links + guards --

def _read_emb(env, atom_id) -> bytes:
    row = env.atoms.db.fetch_one("SELECT HEX(embedding) AS emb FROM atoms WHERE id=%s",
                                 (atom_id,))
    return bytes.fromhex(row["emb"])


def test_remember_writes_atom_with_refs_and_appends_episode(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="原叙事。",
                     evidence=[("Caroline", "展期包含儿童展区")])
    stamped = "[2026-08-20] Caroline 说画展新增儿童展区"          # what stamped_atom_text produces
    d = env.deps(TableEmbedder({stamped: _v(1, 0, 0, 0)}))
    d.reg.ensure(c.id)
    out = tool_remember(d, c="c1", text="Caroline 说画展新增儿童展区",
                        quote="展期包含儿童展区", holder="Caroline", kind="K04",
                        domains=["D13"], episode_append="补充:新增了儿童展区。")
    assert "Written back:" in out and "added 1 index atom (linked to 1 source utterance(s))" in out
    saved = env.atoms.list_by_cell(c.id)
    assert len(saved) == 1 and saved[0].source == "deep"        # the provenance marker
    assert saved[0].kind == "K04" and saved[0].holder == "Caroline"
    assert saved[0].evidence_refs                               # the quote matched verbatim and linked back to the evidence
    assert np.frombuffer(_read_emb(env, saved[0].id),
                         dtype=np.float32).tolist() == [1, 0, 0, 0]   # embedded using the stamped text
    episode = env.cells.get(c.id).episode
    # Appended, never rewritten.
    assert episode.startswith("原叙事。") and episode.endswith("补充:新增了儿童展区。")


def test_remember_quote_miss_leaves_refs_empty_but_writes(db):
    env = Env(db)
    c = env.add_cell(topic="画展", episode="原叙事。",
                     evidence=[("Caroline", "展期包含儿童展区")])
    d = env.deps(TableEmbedder({"[2026-08-20] x": _v(1, 0, 0, 0)}))
    d.reg.ensure(c.id)
    out = tool_remember(d, c="c1", text="x", quote="原话里没有这句")   # no verbatim match
    assert "added 1 index atom (linked to 0 source utterance(s))" in out
    assert env.atoms.list_by_cell(c.id)[0].evidence_refs == []


def test_remember_guards(db):
    env = Env(db)
    c = env.add_cell(topic="t", episode="e")
    d_off = env.deps(ZeroEmbedder(), deep_write=False)
    d_off.reg.ensure(c.id)
    assert "read-only" in tool_remember(d_off, c="c1", text="x")   # the master switch

    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    assert "at least one of text / episode_append" in tool_remember(d, c="c1")   # an empty write
    assert "Cannot write back" in tool_remember(d, c="c7", text="x")   # an unknown handle
    d.remembered = 8
    assert "Write-back cap" in tool_remember(d, c="c1", text="x")   # the per-session cap


# -- MaasChatModel: message mapping + client-side truncation at stop --

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
    # Truncated client-side at the stop sequence, which is what prevents hallucinated continuations.
    assert res.content == "前半"
    assert captured[0]["msgs"] == [{"role": "system", "content": "系统"},
                                   {"role": "user", "content": "问"}]   # role mapping
    assert captured[0]["t"] == 0.1 and captured[0]["m"] == 64    # parameters passed straight through


# -- run_deep: the agent loop (handoff package -> tools -> final answer translated back) --

def test_open_cell_accepts_common_arg_aliases(db):
    """Learned from live runs: the model frequently writes c as cell_id or cell, so schema
    aliases absorb that instead of letting a whole agent round crash."""
    from personos.online.deep_recall import build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。",
                     atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    tools = {t.name: t for t in build_tools(d)}
    assert "叙事全文。" in tools["open_cell"].invoke({"cell_id": "c1"})
    assert "0 utterance(s)" in tools["get_cell_evidence"].invoke({"cell": "c1"})
    # Also learned from live runs: the model guesses parameter names like handle or
    # cell_handle straight from the "Cell handle" wording in the description. Without those
    # aliases, every question wastes a step failing validation and then correcting itself.
    assert "叙事全文。" in tools["open_cell"].invoke({"handle": "c1"})
    assert "0 utterance(s)" in tools["get_cell_evidence"].invoke({"cell_handle": "c1"})


def test_tool_batch_runs_many_calls_in_one_step(db):
    """Passing a list of argument objects as action_input means a batched call in one step:
    each is executed in turn but only one step of budget is charged."""
    from personos.online.deep_recall import _MAX_STEPS, build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。",
                     atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    tools = {t.name: t for t in build_tools(d)}
    out = tools["open_cell"].invoke({"batch": [{"c": "c1"}, {"c": "c1"}]})
    assert "── batch 1/2 ──" in out and "── batch 2/2 ──" in out
    assert out.count("叙事全文。") == 2                          # both calls really ran
    assert d.calls_made == 1                                    # a batch burns only one step
    # The tail of the observation reports the remaining budget.
    assert f"≈{_MAX_STEPS - 1} tool steps left" in out


def test_tool_budget_notice_presses_for_answer_near_cap(db):
    """With 2 or fewer steps of budget left, the observation tail escalates to a "wrap up and
    answer now" prompt."""
    from personos.online.deep_recall import _MAX_STEPS, build_tools
    env = Env(db)
    c = env.add_cell(topic="画展", episode="叙事全文。")
    d = env.deps(ZeroEmbedder())
    d.reg.ensure(c.id)
    d.calls_made = _MAX_STEPS - 3                             # after this call, 2 steps remain
    tools = {t.name: t for t in build_tools(d)}
    out = tools["open_cell"].invoke({"c": "c1"})
    assert "≈2 tool steps left" in out and "budget nearly exhausted" in out


def test_search_tools_accept_singular_aliases(db):
    """Learned from live runs: the model likes the singular keyword and domain, so aliases
    absorb that rather than burning a step on trial and error."""
    from personos.online.deep_recall import build_tools
    env = Env(db)
    env.add_cell(topic="画展", episode="叙事全文。",
                 atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}],
                 evidence=[("user", "周末去看画展")])
    d = env.deps(ZeroEmbedder())
    tools = {t.name: t for t in build_tools(d)}
    out = tools["search_evidence"].invoke({"keyword": "画展"})   # singular name plus a bare string
    assert "周末去看画展" in out
    out = tools["find_cells"].invoke({"domain": ["D01"]})
    assert "Tool argument validation failed" not in out
    out = tools["search_atoms"].invoke({"query": "x", "domain": ["D99"]})
    assert "Tool argument validation failed" not in out


def test_schema_tolerates_junk_scalar_args(db):
    """Learned from live runs: the model occasionally passes page as "abc", so the schema layer
    tolerates it and falls back to the default instead of crashing the loop."""
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
    # The junk page fell back to 1 and the tool returned normally.
    assert "t" in out.steps[0]["obs_head"]


def test_tool_exception_becomes_observation_not_crash(db):
    """handle_tool_error: an exception inside a tool function (here the embedder is down)
    becomes observation text, so the agent loop is not interrupted."""
    class BoomEmbedder:
        def embed(self, _texts):
            raise RuntimeError("embed 服务不可用")

    env = Env(db)
    # The pool has to be non-empty for execution to reach the embed call.
    env.add_cell(topic="t", episode="e", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    llm = FakeLLM([
        '```json\n{"thought": "先检索", "action": "search_atoms", "action_input": {"query": "q"}}\n```',
        '```json\n{"thought": "检索挂了,直接作答", "action": "Final Answer",'
        ' "action_input": {"answer": "答", "cited": []}}\n```',
    ])
    out = run_deep(llm, BoomEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    assert out.ans.answer == "答" and len(out.steps) == 1
    # The exception turned into an observation, so the agent can correct itself.
    assert "embed 服务不可用" in out.steps[0]["obs_head"]


def test_wrong_field_name_becomes_observation_not_crash(db):
    """A completely wrong field name (target, which is neither c nor an alias) turns the
    validation error into an observation hint, so the agent loop is not interrupted."""
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
    # c9 is unknown and gets dropped; c1 is translated back into the real id.
    assert out.ans.cited_cells == [c.id]
    assert [(s["tool"], s["args"]["query"]) for s in out.steps] == [("search_atoms", "画展 展期")]
    for part in ("## Task", "## Memory catalog", "## Current time"):   # the handoff sections are all present
        assert part in out.handoff
    assert out.secs["deep"] >= 0


def test_run_deep_iteration_cap_yields_empty_answer(db):
    env = Env(db)
    env.add_cell(topic="t", episode="e", atoms=[{"text": "x", "vec": _v(1, 0, 0, 0)}])
    loop = ('```json\n{"thought": "再翻一页", "action": "open_cell", "action_input": {"c": "c1"}}\n```')
    llm = FakeLLM([loop] * 20)                                  # never stops, so the 9-step cap trips
    out = run_deep(llm, ZeroEmbedder(), env.atoms, env.cells, env.ev,
                   query="q", now_dt=_T, deep_write=False)
    # Honestly reports "no answer" rather than inventing one.
    assert out.ans.answer == "" and len(out.steps) == 9


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
    # (1) The task.
    assert "画展展期?" in handoff and "question subject: Caroline" in handoff
    assert "2026-08-01 ~ 2026-08-31" in handoff and "D13" in handoff
    # (2) The full text of the fast-path evidence.
    assert "画展叙事。" in handoff and "━━━ c1 ━━━" in handoff
    # (3) The adjudication and the gap statement, verbatim.
    assert "the retrieved materials lack" in handoff and "缺画展的具体展期日期" in handoff
    assert "跑步" in handoff and "0 older cells exist" in handoff        # (4) the catalog
    assert "## Current time" in handoff                                 # (5) the current time
    # No answer draft is handed over.
    assert "作答" not in handoff or "草稿" not in handoff


def test_handoff_translates_fast_window_handles_to_deep_handles(db):
    """The two handle schemes are independent: the adjudication critique writes fast-path mN
    handles (the material window order), and the handoff translates them into deep-track cN
    handles (the catalog registration order, where c1 is the newest). The materials section is
    rendered with deep-track handles, so gap references have to line up with it.

    Out-of-range handles are kept verbatim: a visibly unfamiliar symbol is better than silently
    pointing at the wrong cell."""
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
    # Deep-track handles: venue becomes c2 in catalog order.
    assert "━━━ c2 ━━━" in handoff and "场地叙事。" in handoff
    # m1 maps to venue = c2 and m2 to prep = c3, the same scheme the materials section uses.
    assert "(c2, c3)" in handoff
    assert "m1" not in handoff and "m2" not in handoff
    assert "m9 is bogus" in handoff       # out of range: kept verbatim, no mapping invented


# -- The run_recall fork (assertions at the orchestration layer) --

def test_run_recall_deep_mode_bypasses_fast(db, evidence_store):
    from personos.online.recall_flow import run_recall
    env = Env(db)
    c = env.add_cell(topic="画展", episode="画展叙事。",
                     atoms=[{"text": "展期 2026-09", "vec": _v(1, 0, 0, 0)}])
    emb = TableEmbedder({"画展筹备": _v(1, 0, 0, 0), "展期": _v(1, 0, 0, 0)})
    state = {"agent": 0}

    class DeepLLM:
        """R0 goes through the query preprocessor; when the deep-track system wording matches,
        return an agent JSON block. Any other station means it should not have been called."""

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
            raise AssertionError(f"deep mode should not call this station: {sys[:40]!r}")

    o = run_recall(DeepLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="deep")
    assert o.hits == [] and o.reviews == []                      # every fast-path station is skipped
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
            # The reviewer check comes before answerer because that word also appears in the
            # adjudication prompt.
            if "answer reviewer" in sys:
                return '{"verdict":"insufficient_material","critique":"缺展期日期"}'
            if "answerer" in sys:
                return '{"answer":"(快链的凑合答案)","cells":["c1"]}'
            if "deep-retrieval agent" in sys:
                return ('```json\n{"thought": "直接终答", "action": "Final Answer",'
                        ' "action_input": {"answer": "展期在 2026-09。", "cited": ["c1"]}}\n```')
            raise AssertionError(f"unknown station: {sys[:40]!r}")

    o = run_recall(EscalateLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="auto")
    assert o.escalated and o.deep is not None
    # Insufficient material does not trigger a re-answer; it escalates straight to the deep track.
    assert not o.retried
    # The critique is handed to the deep track as the direction of the gap.
    assert "缺展期日期" in o.deep.handoff
    assert o.ans.answer == "展期在 2026-09。"                    # the deep track's final answer overrides the fast path
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
            # The reviewer check comes before answerer because that word also appears in the
            # adjudication prompt.
            if "answer reviewer" in sys:
                return '{"verdict":"ok","critique":""}'
            if "answerer" in sys:
                return '{"answer":"展期在 2026-09。","cells":["c1"]}'
            raise AssertionError(f"an ok verdict must not escalate to the deep track: {sys[:40]!r}")

    o = run_recall(OkLLM(), emb, env.atoms, env.cells, evidence_store,
                   session_id="s1", query="画展什么时候", now_dt=_T, mode="auto")
    assert not o.escalated and o.deep is None and not o.retried
    assert o.ans.answer == "展期在 2026-09。"
