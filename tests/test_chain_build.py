"""W2.5 chain-assignment tests: pre-filter, group adjudication, persistence, plus the
conservative rule and non-blocking failure (docs/atom-chain-design.md §4.2).

No real model provider is called: RoutingLLM routes on the system prompt, and the chain
assigner calls are fed pre-set groupings.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from personos.models import ChainInfo, EvidenceRecord, MemoryAtom
from personos.online.chain_build import assign_chains
from personos.online.write_path import build_cell
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.evidence_store import EvidenceStore

from .fakes import FakeEmbedder

_T0 = datetime(2026, 8, 25, 10, 0)


class RoutingLLM:
    """The three kinds of write-path call plus the chain assigner, dispatched four ways;
    records the call order."""

    def __init__(self, episode=(), atoms=(), assign=()):
        self.q = {"episode": list(episode), "atoms": list(atoms), "assign": list(assign)}

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        sysp = messages[0]["content"]
        kind = ("episode" if "episode weaver" in sysp
                else "atoms" if "atomic-memory extractor" in sysp
                else "assign" if "chain assigner" in sysp
                else "boundary" if "boundary detector" in sysp else "other")
        assert kind != "other", f"unknown system prompt: {sysp[:40]}"
        if kind == "boundary":
            return '{"should_end": true, "confidence": 0.9, "topic_summary": "t"}'
        assert self.q[kind], f"unexpected {kind} call (the queue is empty)"
        resp = self.q[kind].pop(0)
        return resp(messages[-1]["content"]) if callable(resp) else resp


def _items(texts: list[str]):
    """Atoms plus their vectors (fixed seed, so the cosines are reproducible)."""
    rng = np.random.default_rng(7)
    vecs = [rng.standard_normal(8).astype(np.float32) for _ in texts]
    out = []
    for t, v in zip(texts, vecs):
        a = MemoryAtom(memcell_id="c1", text=t, object_type="fact", occurrence_time=_T0)
        out.append((a, v))
    return out


def _seeded(db, texts: list[str]):
    """Items with W2 semantics: persist first (chain assignment presupposes the atoms already
    exist), then return the (atom, vec) pairs."""
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


# -- Chain assignment proper --

def test_new_chain_grouping(db):
    """Starting from zero chains: the LLM groups all three atoms into one new chain, members
    keep their within-cell order, and the centroid is the group mean."""
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
    """Appending to an existing chain: the pre-filtered candidates make it into the prompt, the
    append is persisted, and the centroid is updated incrementally at group level."""
    cs = ChainStore(db)
    old_vec = rng_vec(1)
    ch0, _ = _chain_with_atom(db, "Caroline 的瑜伽馆", "Caroline 的瑜伽馆在 MBS", old_vec)

    items = _seeded(db, ["Caroline 的瑜伽馆搬到了 Marina Bay(2026-08-20)",
                         "Caroline 每周三练热瑜伽"])
    # Candidate chain labels follow chain creation order; the callable picks the target label
    # back out of the prompt.
    def pick(prompt):
        label = next(candidate for candidate in ("c1", "c2") if f"{candidate} · Caroline 的瑜伽馆" in prompt)
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
    """A hallucinated chain label leaves the affected atom free: no chain is created and
    nothing is appended (the conservative rule — better to miss than to be wrong)."""
    cs = ChainStore(db)
    _chain_with_atom(db, "已有链", "已有事实",
                     np.random.default_rng(3).standard_normal(8).astype(np.float32))
    items = _seeded(db, ["一个新事实"])
    llm = RoutingLLM(assign=['{"assignments":[{"chain":"c9","atoms":[1]}]}'])
    res = assign_chains(llm, cs, items)
    assert res.free == [items[0][0].id] and not res.new_chains and not res.appended


def test_parse_failure_all_free(db):
    """Non-JSON output from the LLM leaves everything free and raises nothing — failure here
    must not block the write path."""
    cs = ChainStore(db)
    items = _seeded(db, ["事实A", "事实B"])
    llm = RoutingLLM(assign=["not-json", "still-not-json"])   # both num_tries=2 attempts are bad
    res = assign_chains(llm, cs, items, origin_cell_id="c")
    assert sorted(res.free) == sorted(a.id for a, _ in items)
    assert cs.list_chains() == []


def test_unplaced_atoms_default_free(db):
    """Atoms the LLM omitted default to free: every atom must be placed explicitly, and an
    omission means a new chain or free — never a guess."""
    cs = ChainStore(db)
    items = _seeded(db, ["事实A", "事实B", "事实C"])
    llm = RoutingLLM(assign=['{"assignments":[{"chain":"new","title":"一组","atoms":[1]}]}'])
    res = assign_chains(llm, cs, items)
    assert len(res.free) == 2 and not any(
        a.id in res.free for a, _ in items[:1])


def test_execution_conflict_isolated(db):
    """When one group fails to execute (a double-attachment conflict), only that group is lost;
    the remaining groups are persisted as usual."""
    cs = ChainStore(db)
    ch_b, b1 = _chain_with_atom(db, "链B", "B 的事实", np.random.default_rng(5).standard_normal(8).astype(np.float32))
    fresh = _seeded(db, ["干净的新事实"])
    items = [(b1, np.random.default_rng(6).standard_normal(8).astype(np.float32))] + fresh
    # The grouping: atom1 (already on chain B) is pointed at chain B for an append, which
    # conflicts with expect_chain; atom2 opens a new chain and should succeed.
    llm = RoutingLLM(assign=[
        lambda p: json.dumps({"assignments": [
            {"chain": "c1", "atoms": [1]},
            {"chain": "new", "title": "新链", "atoms": [2]}]})])
    res = assign_chains(llm, cs, items)
    assert len(res.free) == 1 and len(res.new_chains) == 1
    assert cs.get_chain(ch_b.id).n_atoms == 1          # chain B was not contaminated
    assert cs.full_chain(res.new_chains[0].id)[0].text == "干净的新事实"


# -- build_cell wiring --

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
    """The LLM cannot place anything (assignments is empty): every atom stays free and the
    write path is not blocked."""
    llm = RoutingLLM(episode=[_EPISODE_OK], atoms=[_ATOMS_OK],
                     assign=['{"assignments": []}'])
    cb = _build(db, llm)
    assert cb.chain_assign is not None
    assert all(a.chain_id == "" for a in cb.atoms)


def test_build_cell_chains(db):
    """W2.5 runs after W2 persistence, and the two attributes each form their own chain — the
    foundation for splitting attributes into separate chains."""
    llm = RoutingLLM(episode=[_EPISODE_OK], atoms=[_ATOMS_OK], assign=[_ASSIGN_OK])
    cb = _build(db, llm)
    assert cb.chain_assign is not None
    assert len(cb.chain_assign.new_chains) == 2 and not cb.chain_assign.free
    titles = {c.title for c in cb.chain_assign.new_chains}
    assert titles == {"Caroline 的瑜伽频率", "Caroline 的瑜伽地点"}
    by_text = {a.text: a.chain_id for a, _ in AtomStore(db).all_with_embeddings()}
    assert by_text["Caroline 每周三练热瑜伽"] != by_text["Caroline 的瑜伽馆在 MBS"]
