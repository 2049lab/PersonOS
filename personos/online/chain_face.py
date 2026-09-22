"""Unit assembly: the R1 atom pool -> the material-unit station.

One rule: for each atom in the pool, look up its chain — a chain with >=2 nodes goes to the weaver
(S4), which weaves it into a memcell' (the LLM sees only query + chain title + atom checklist +
episodes, and uses the query to set emphasis); a single-node chain or a free-floating atom maps to
the memcell it lives in (no LLM). Dedup by chain/cell (several atoms pointing at the same chain or
the same cell still yield one unit).

A memcell' is a temporary view: structurally identical to a plain unit (a CellHit with topic = chain
title, episode = woven text, time = the span of its member cells), and `covers` only carries the
short/long handle mapping (mN -> member cells), so everything downstream (rerank, answering,
adjudication) is completely unaware of the difference. The chain face is purely derived: no chain or
a failed weave always degrades to cell-granularity plain units (that chain's atoms go back to their
own cell buckets) and never blocks the main read path.
"""

from __future__ import annotations

import os
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from .. import obs
from ..models import ChainInfo, MemCell, ensure_aware
from ..online.retrieval import AtomHit, CellHit
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from .llm import ChatLLM

_BOUNDARY_TITLES = 5   # max number of out-of-pool chain titles named in the boundary note (prompt economy)
_MAX_EPISODES = 15     # max member cells used as weaving input (beyond that, take the most recent 15 and mark the truncation at the top of the input)

# Max chains woven in parallel within one recall (one LLM call per chain, IO-bound; latency is about
# +1 segment).
# Note: this is the per-REQUEST fan-out width, not a global pool — worst-case provider concurrency equals
# recall concurrency x this value.
_WEAVE_WORKERS = int(os.environ.get("PERSONOS_WEAVE_WORKERS", "8"))
_SPAN_FLOOR = datetime(1, 1, 1, tzinfo=timezone.utc)   # sort floor for a missing t_start (timezone-aware)


@dataclass
class UnitAssembly:
    """What one unit assembly produced: material units + boundary note + inspection counters."""
    units: list[CellHit] = field(default_factory=list)   # pool order (position of the chain's/cell's first appearance)
    boundary: str = ""        # generic boundary note (no LLM; feeds the BOUNDARY section of R5/R3' directly, D-C10)
    n_pool: int = 0           # number of atoms in the R1 pool
    n_chains: int = 0         # number of >=2-node chains the pool hit (i.e. number of weave tasks)
    n_woven: int = 0          # how many weaves succeeded
    n_plain: int = 0          # number of plain units (including cells that came back from a failed weave)


def assemble_units(
    pool: list[AtomHit], chains: ChainStore, cells: CellStore,
    llm: ChatLLM | None, *, query: str,
    beyond: list[AtomHit] | None = None,
) -> UnitAssembly:
    """Pool atoms -> material units (woven memcell' + plain units) + boundary note.

    chains: source of chain metadata (title / n_atoms); cells: source of the cells behind plain units
    and weaving input; llm: the weaver (for >=2-node chains; None means everything degrades to plain
    units, used in tests and in degraded mode); beyond: matching atoms outside the R1 pool (the
    boundary note names their chain titles). Any assembly error degrades to all-plain units; nothing
    is raised.
    """
    asm = UnitAssembly(n_pool=len(pool))
    try:
        _assemble(asm, pool, chains, cells, llm, query=query, beyond=beyond or [])
    except Exception as e:   # noqa: BLE001  assembly failed: fall back to cell granularity (chain atoms go back to their own cells too); never raise
        logger.warning(f"unit assembly failed, degrading to all-plain units: {e}")
        asm.units = _plain_units(pool, cells)
        asm.boundary = ""
        asm.n_chains = asm.n_woven = 0
        asm.n_plain = len(asm.units)
    unit_detail = "\n".join(
        f"    [{i}] topic={u.cell.topic!r} atoms={len(u.atoms)}\n"
        f"      episode/woven_text={u.cell.episode!r}" for i, u in enumerate(asm.units, 1))
    logger.info(f"unit assembly pool={asm.n_pool} chains={asm.n_chains} woven={asm.n_woven} "
                f"plain={asm.n_plain} boundary={asm.boundary!r} q={query!r}\n"
                f"  material units(n={len(asm.units)}):\n{unit_detail}")
    return asm


def _assemble(asm: UnitAssembly, pool: list[AtomHit], chains: ChainStore,
              cells: CellStore, llm, *, query: str, beyond: list[AtomHit]) -> None:
    # 1) Bucket: each pool atom goes to a chain bucket (>=2-node chain) or a cell bucket
    #    (free-floating / single-node chain / dangling chain_id)
    infos: dict[str, ChainInfo] = {c.id: c for c, _ in chains.list_chains()}
    chain_atoms: dict[str, list[AtomHit]] = {}   # chain_id -> matching atoms in the pool (pool order)
    plain_atoms: dict[str, list[AtomHit]] = {}   # cell_id -> matching atoms in the pool (pool order)
    for ah in pool:
        info = infos.get(ah.atom.chain_id or "")
        if info is not None and info.n_atoms >= 2:
            chain_atoms.setdefault(info.id, []).append(ah)
        elif ah.atom.memcell_id:
            plain_atoms.setdefault(ah.atom.memcell_id, []).append(ah)
    asm.n_chains = len(chain_atoms)

    # 2) Weave (one LLM call per >=2-node chain; on failure or absence that chain's atoms go back to
    #    the cell buckets, so no information is lost)
    woven = _weave_all(llm, query, chain_atoms, infos, chains, cells)
    for cid, hits in chain_atoms.items():
        if cid not in woven:
            for ah in hits:
                plain_atoms.setdefault(ah.atom.memcell_id, []).append(ah)

    # 3) Unit order: follow the pool order by first appearance — a chain unit appears at the position
    #    of its first member. When a cell holds both free-floating atoms and chain atoms, the plain
    #    unit and the woven unit are both kept (the cell episode and the woven chain text are at
    #    different granularities and do not substitute for each other)
    units: list[CellHit] = []
    emitted: set[tuple[str, str]] = set()
    for ah in pool:
        a = ah.atom
        key = (("chain", a.chain_id) if a.chain_id in woven
               else ("cell", a.memcell_id))
        if key in emitted:
            continue
        if key[0] == "chain":
            emitted.add(key)
            units.append(woven[a.chain_id])
        else:
            cell = cells.get(a.memcell_id)
            if cell is None:
                continue                                   # orphaned atom (its cell was deleted): skip instead of crashing
            emitted.add(key)
            units.append(_plain_unit(cell, plain_atoms[a.memcell_id]))
    asm.units = units
    asm.n_woven = sum(1 for u in units if u.covers)
    asm.n_plain = len(units) - asm.n_woven

    # 4) Generic boundary note (D-C10): among the out-of-pool atoms, count only those whose fact is
    #    genuinely absent from the materials — out-of-pool members of an already-woven chain (the
    #    weave covers the whole chain) and atoms whose cell is already in the materials do not count
    #    as missing, otherwise downstream gets a fake "material is missing" signal that misleads the
    #    count/enumeration judgment in R5 and R3'
    covered = {cid for u in units for cid in (u.covers or [u.cell.id])}
    missing = [ah for ah in beyond
               if (ah.atom.chain_id or "") not in woven
               and ah.atom.memcell_id not in covered]
    asm.boundary = _boundary_note(missing, infos)


def _plain_units(pool: list[AtomHit], cells: CellStore) -> list[CellHit]:
    """Fallback when assembly fails: plain units for the whole pool, deduped by cell (chain atoms go
    back to their own cells too, ordered by first appearance in the pool)."""
    per_cell: dict[str, list[AtomHit]] = {}
    order: list[str] = []
    for ah in pool:
        cid = ah.atom.memcell_id
        if not cid:
            continue
        if cid not in per_cell:
            per_cell[cid] = []
            order.append(cid)
        per_cell[cid].append(ah)
    out = []
    for cid in order:
        cell = cells.get(cid)
        if cell is not None:
            out.append(_plain_unit(cell, per_cell[cid]))
    return out


def _plain_unit(cell: MemCell, hits: list[AtomHit]) -> CellHit:
    """Plain unit: the pool atoms that hit this cell (descending similarity); its score is the best
    atom in the pool."""
    atoms = sorted(hits, key=lambda h: -h.similarity)
    best = max(hits, key=lambda h: h.rrf)
    return CellHit(cell=cell, score=best.rrf, best_sim=best.similarity, atoms=atoms)


# -- Weaving (§5.3, D-C4): parallel via a thread pool, one LLM call per chain --

def _weave_all(llm, query: str, chain_atoms: dict[str, list[AtomHit]],
               infos: dict[str, ChainInfo], chains: ChainStore,
               cells: CellStore) -> dict[str, CellHit]:
    """Weave every >=2-node chain -> {chain_id: woven unit}. A chain that fails is absent from the
    return value (the caller degrades it).

    The current stack is synchronous and does not pull in asyncio — query-side latency is about
    +1 segment, not +N. All DB reads happen serially on the calling thread: the shared connection is
    not thread-safe and concurrent reads scramble pymysql's protocol state. A single failing chain
    only degrades that chain (WARNING) and never blocks the main read path.
    """
    if llm is None or not chain_atoms:
        return {}
    prepped = []          # (chain_id, members, member_cells, episodes, truncated) -- gather the input serially first
    for chain_id in chain_atoms:
        try:
            members, member_cells, episodes, trunc = _gather(chains, cells, infos[chain_id])
            prepped.append((chain_id, members, member_cells, episodes, trunc))
        except Exception as e:   # noqa: BLE001  failed to gather input: degrade this chain, do not hold up the rest
            logger.warning(f"failed to gather weaving input, chain «{infos[chain_id].title}» "
                           f"degrades to plain member-cell units: {e}")
    out: dict[str, CellHit] = {}
    if not prepped:
        return out
    # Context-preserving thread pool: makes the chat calls of the parallel weaves nest correctly
    # under the recall root trace (otherwise they fly off as orphan root traces)
    with obs.ContextThreadPoolExecutor(max_workers=min(_WEAVE_WORKERS, len(prepped))) as ex:
        futs = {ex.submit(weave_chain, llm, query=query, chain=infos[chain_id],
                          members=members, episodes=episodes, truncated=trunc):
                (chain_id, member_cells)
                for chain_id, members, member_cells, episodes, trunc in prepped}
        for fut in as_completed(futs):
            chain_id, member_cells = futs[fut]
            try:
                out[chain_id] = _synth_unit(infos[chain_id], member_cells,
                                            chain_atoms[chain_id], fut.result())
            except Exception as e:   # noqa: BLE001  weave failed: degrade this chain, do not hold up the rest
                logger.warning(f"weaving failed, chain «{infos[chain_id].title}» "
                               f"degrades to plain member-cell units: {e}")
    return out


def _gather(chains: ChainStore, cells: CellStore, info: ChainInfo):
    """Weaving input for one chain (executed serially on the calling thread): the whole chain's atoms
    (in chain order) plus the full set of member cells.

    `episodes` holds only the most recent <=15 cells (beyond that the truncation is marked at the top
    of the input); the atom checklist and `covers` use the full set — the former is the coverage
    self-check scope, the latter the scope for expanding citations (§5.2).
    """
    members = chains.full_chain(info.id)       # pull the whole chain, in chain order
    cell_by_id: dict[str, MemCell] = {}
    for a in members:                          # dedup member cells while preserving order (chain order)
        if a.memcell_id and a.memcell_id not in cell_by_id:
            c = cells.get(a.memcell_id)
            if c is not None:
                cell_by_id[a.memcell_id] = c
    member_cells = list(cell_by_id.values())
    shown = sorted(member_cells, key=lambda c: ensure_aware(c.t_start) or _SPAN_FLOOR)
    trunc = None
    if len(shown) > _MAX_EPISODES:
        trunc = (_MAX_EPISODES, len(shown))
        shown = shown[-_MAX_EPISODES:]                    # the most recent 15 cells
    return members, member_cells, [(c, c.episode or "") for c in shown], trunc


def _synth_unit(info: ChainInfo, cells: list[MemCell],
                hits: list[AtomHit], woven: str) -> CellHit:
    """Woven text -> woven unit (cell.id = chain id, topic = chain title, episode = woven text,
    covers = the full set of member cells)."""
    starts = [t for t in (ensure_aware(c.t_start) for c in cells) if t]
    ends = [t for t in (ensure_aware(c.t_end) for c in cells) if t]
    atoms = sorted(hits, key=lambda h: -h.similarity)
    best = max(hits, key=lambda h: h.rrf)
    return CellHit(
        cell=MemCell(id=info.id, topic=info.title, episode=woven,
                     t_start=min(starts) if starts else None,
                     t_end=max(ends) if ends else None),
        score=best.rrf, best_sim=best.similarity,    # relevance is represented by the best member in the pool
        atoms=atoms, covers=[c.id for c in cells])


def _boundary_note(missing: list[AtomHit], infos: dict[str, ChainInfo]) -> str:
    """D-C10 generic boundary note (no LLM): there are matching atoms outside the pool that the
    materials do not cover; name the titles of their >=2-node chains.

    The argument is already filtered: members of an already-woven chain and atoms whose cell is
    already in the materials do not count as missing (see step 4 of _assemble).
    """
    if not missing:
        return ""
    titles: list[str] = []
    for ah in missing:
        info = infos.get(ah.atom.chain_id or "")
        if (info is not None and info.n_atoms >= 2 and info.title
                and info.title not in titles):
            titles.append(info.title)
            if len(titles) >= _BOUNDARY_TITLES:
                break
    titled = f" (chains: {', '.join(titles)})" if titles else ""
    return (f"{len(missing)} more matching atoms exist beyond the shown materials{titled}. "
            "The shown materials may not cover the full set — enumerate what is shown "
            "and state the boundary when the list cannot be completed.")


# -- The weaver (§5.3, D-C4) --

_WEAVE_SYSTEM = """# Role
You are the chain weaver in a memory read pipeline: ONE fact-chain holds the statements a user made
about the same matter over time (oldest → newest). Weave its member segments' episodes into ONE
coherent narrative that will serve as answer material.

# Input
- Query: the current question — it decides EMPHASIS only (what to expand vs compress), never what
  to keep.
- Chain title, and the chain's atoms in chain order, each with its date.
- Source episodes of the member segments — the ONLY fact source.

# Hard rules
1. Integrate, never adjudicate. Old and new statements are BOTH kept, each with its own date.
   A "correction" may be written only when the dialogue itself explicitly corrected an earlier
   statement (render it as: first said X, later corrected to Y). You never pick a winner —
   downstream answer-time resolution is the designed fallback; do not preempt it.
2. Coverage first. Before writing, walk the atom checklist: EVERY atom's fact must be traceable
   in your narrative. A missing chain fact is a defect.
3. Episodes are the only fact source. Anything an atom claims that NO source episode contains must
   not be written (extraction noise must not become hallucination). Better to omit one sentence
   than to invent.
4. Emphasis = detail level, not selection. Expand what the query touches, compress the rest to a
  clause — but no chain fact is dropped.
5. Language follows the source episodes. Third-person narration. Time double-annotated: every
   dated statement carries its absolute date inline (e.g. 2026-08-10).

# Output
The woven narrative ONLY — no preamble, no headings, no commentary. Episode-style prose."""


def weave_chain(llm: ChatLLM, *, query: str, chain: ChainInfo, members: list,
                episodes: list[tuple[MemCell, str]],
                truncated: tuple[int, int] | None = None) -> str:
    """Weave one chain: the whole chain's atom checklist + the member cells' episodes as raw input ->
    one woven passage (in episode format).

    members: every atom of the chain (chain order, each with its occurrence_time); episodes: a list
    of (member cell, its episode); truncated: (cells actually shown, total member cells) — set when
    more than 15 cells were cut down to the most recent ones. Empty woven text counts as a failure
    (raises ValueError).
    """
    atom_lines = []
    for a in members:
        when = a.occurrence_time.strftime("%Y-%m-%d") if getattr(a, "occurrence_time", None) else ""
        atom_lines.append(f"[{when}] {a.text}")
    ep_lines = []
    for c, ep in episodes:
        ep_lines.append(f"—— segment {c.id} · topic: {c.topic or '(no topic)'} ——\n{ep or '(empty)'}")
    # Resolve the truncation: the atom checklist is the full set while `episodes` only shows the most
    # recent 15 cells — atoms belonging to a cut segment are written only as far as the shown
    # episodes support them, and silently omitted otherwise, which does not count as a coverage
    # defect (this closes the contradiction between Hard rule 2 and Hard rule 3)
    cut = (f"\n(showing the most recent {truncated[0]} of {truncated[1]} segments; earlier ones "
           f"truncated — atoms dating from those hidden segments count as covered when the shown "
           f"episodes support them; otherwise omit them silently, that omission is not a defect)\n"
           if truncated else "\n")
    user = (f"—— Query ——\n{query}\n\n"
            f"—— Chain: {chain.title or '(untitled)'} ——\n"
            f"atoms (oldest → newest), the coverage checklist:\n" + "\n".join(atom_lines)
            + f"\n\n—— Source episodes ({len(episodes)} segments) ——{cut}" + "\n".join(ep_lines))
    with obs.stage("weave_chain"):
        text = llm.chat([{"role": "system", "content": _WEAVE_SYSTEM},
                         {"role": "user", "content": user}], temperature=0.2, max_tokens=900).strip()
    logger.info(f"LLM[weave_chain] chain«{chain.title}»\n  -- input(user) --\n{user}\n  -- output(woven) --\n{text}")
    if not text:
        raise ValueError("weaving returned empty text")
    return text
