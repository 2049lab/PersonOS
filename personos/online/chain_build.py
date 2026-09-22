"""W2.5 chain assignment: put each new atom onto an existing chain
or start a new one.

Runs right after build_cell's upsert_many, one batched LLM call per cell:
prefilter (cosine between atom vector and chain centroid, top-5 per atom) -> one LLM grouping
decision -> persist.

The conservative law is a hard rule: when in doubt, start a new chain. A missed link only falls back
to the status quo (retrieval oversampling can still save it); a wrong link is poison (chain expansion
drags an unrelated fact into the material). The assigner only appends — it never merges chains, moves
members, or deletes them.

Failures are non-blocking: an LLM parse failure or timeout leaves every atom of this cell free-floating
and logs a WARNING; a chain is a derived index and must never block the main write path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from ..models import ChainInfo, MemoryAtom
from ..storage.chain_store import ChainStore
from .llm import ChatLLM, chat_json

_CANDIDATES_PER_ATOM = 5   # prefilter: max candidate chains per atom (all of them when there are fewer)
_TAIL_CONTEXT = 8          # how many members from the tail of a candidate chain the LLM sees (recent state of that chain)


@dataclass
class ChainAssignResult:
    """What chain assignment produced for one cell (for workbench inspection and test assertions)."""
    origin_cell_id: str = ""
    new_chains: list[ChainInfo] = field(default_factory=list)
    appended: dict[str, int] = field(default_factory=dict)   # chain_id -> number appended from this cell
    free: list[str] = field(default_factory=list)            # ids of atoms left free-floating

    @property
    def assigned(self) -> int:
        return sum(self.appended.values()) + sum(c.n_atoms for c in self.new_chains)


_ASSIGN_SYSTEM = """# Role
You are the chain assigner in a memory write pipeline: for each newly extracted atom from one dialogue
segment, decide whether it continues an EXISTING chain about the same matter, or starts a NEW chain.

# What a chain is
A chain groups atoms describing the SAME underlying thing evolving over time — one subject + ONE
aspect. E.g. the matter "where Caroline practices yoga": "does hot yoga at gym X" + "her yoga studio
moved to MBS" + "the new studio is closer to her office" are all statements of that one matter.
Atoms about different aspects never share a chain even when topically adjacent: the weekly SCHEDULE
of the yoga, the VENUE of the yoga, and the COACH of the yoga are three different chains. A venue
atom never joins a schedule chain; a one-off plan to try something is not a statement about the
venue either.

# Decision rule (conservative by design)
- Append to an existing chain ONLY when the atom is clearly another statement of that chain's same
  matter (same subject + same aspect). When in doubt → NEW chain.
- A miss (should have joined but didn't) is cheap: retrieval oversampling still surfaces both chains.
  A wrong join is poison: chain expansion drags an unrelated fact into the material.
- Never merge two existing chains, never move or remove members. Append only.
- Group several new atoms into ONE new chain ONLY when they are statements of the same one matter
  (same subject + same aspect). "Her yoga studio moved to MBS" and "MBS offers hot yoga" and "she
  plans to try it next week" are THREE matters (venue / what the venue offers / her plan) → three
  chains. When unsure whether two atoms share a matter → separate chains.
- Every atom must appear in exactly one group; atoms you cannot confidently place go to new chain(s).

# Output (JSON only)
{"assignments":[{"chain":"c2","atoms":[1,4]},{"chain":"new","title":"Caroline's yoga venue","atoms":[2,3]}]}
"chain" is a candidate label (c1, c2, ...) or "new". Title: one short phrase naming the matter
(subject + aspect), in the same language as the atoms."""


def _prefilter(items: list[tuple[MemoryAtom, np.ndarray]],
               chains: list[tuple[ChainInfo, np.ndarray | None]],
               ) -> dict[int, list[int]]:
    """Take the top-5 candidate chains per atom by centroid cosine (returns atom index -> list of
    indices into `chains`).

    Pure numpy, no LLM. Chains without a centroid stay out of the prefilter (they come back on their
    own once recompute fills the centroid in).
    """
    usable = [(i, c) for i, (_, c) in enumerate(chains) if c is not None]
    per: dict[int, list[int]] = {}
    for k, (_, vec) in enumerate(items):
        # Keep the cosine as a float for sorting: cosine lives in [-1,1], so any rounding collapses
        # it to 0 and the sort degenerates into ordering by index
        scores = [(float(np.dot(vec, c) / (np.linalg.norm(vec) * np.linalg.norm(c) + 1e-9)), i)
                  for i, c in usable]
        scores.sort(reverse=True)
        per[k] = [i for _, i in scores[:_CANDIDATES_PER_ATOM]]
    return per


def _render_candidates(chains: list[ChainInfo], idxs: list[int], store: ChainStore) -> str:
    """Render a candidate chain: label + title + the text of the last <=8 members at the tail (the
    recent state of that chain, which is what the join decision is based on)."""
    blocks = []
    for n, i in enumerate(idxs, start=1):
        info = chains[i]
        members = store.full_chain(info.id)[-_TAIL_CONTEXT:]
        lines = [f"    - {a.text}" for a in members] or ["    - (chain has no reachable members)"]
        blocks.append(f"c{n} · {info.title or '(untitled)'} · members (oldest → newest):\n"
                      + "\n".join(lines))
    return "\n".join(blocks)


def assign_chains(llm: ChatLLM, store: ChainStore,
                  items: list[tuple[MemoryAtom, np.ndarray]], *, origin_cell_id: str = "",
                  ) -> ChainAssignResult:
    """Entry point for assigning the new atoms of one cell. Any failure only degrades (atoms are left
    free-floating); nothing is raised."""
    res = ChainAssignResult(origin_cell_id=origin_cell_id)
    if not items:
        return res
    # Order by chain creation time so the c1..cN labels are stable (the chain SELECT is unordered, so
    # without sorting the labels drift from call to call)
    chains = sorted(store.list_chains(), key=lambda p: (p[0].created_at.isoformat(), p[0].id))
    per = _prefilter(items, chains)
    union = sorted({i for cands in per.values() for i in cands})
    label2idx = {f"c{n}": i for n, i in enumerate(union, start=1)}
    idx_by_id = {chains[i][0].id: i for i in union}

    atom_lines = []
    for k, (a, _) in enumerate(items, start=1):
        when = a.occurrence_time.strftime("%Y-%m-%d") if a.occurrence_time else ""
        cands = [f"c{n}" for n, i in enumerate(union, start=1) if i in per[k - 1]]
        cs = f"  candidates: {', '.join(cands)}" if cands else "  candidates: (none — must start new)"
        atom_lines.append(f"[{k}] ({when}) {a.text}\n{cs}")
    cand_block = (_render_candidates([info for info, _ in chains], union, store)
                  if union else "(no existing chains yet — every atom starts a new chain)")
    user = (f"—— New atoms from this segment ——\n" + "\n".join(atom_lines)
            + f"\n\n—— Candidate chains (nearest by meaning) ——\n{cand_block}")

    try:
        data, raw = chat_json(llm, [{"role": "system", "content": _ASSIGN_SYSTEM},
                                    {"role": "user", "content": user}], max_tokens=1500, num_tries=2,
                              stage="chain_assign")
        groups = _parse_groups(data.get("assignments"), len(items))
    except Exception as e:   # noqa: BLE001  a failed assignment must not block the write: leave this whole cell free-floating
        logger.warning(f"W2.5 chain assignment failed, leaving this cell's {len(items)} atoms unchained: {e}")
        res.free = [a.id for a, _ in items]
        return res

    _execute(store, items, groups, label2idx, chains, res)
    id2text = {a.id: a.text for a, _ in items}
    groups_detail = "; ".join(
        f"{label}«{title or '?'}»→{[items[k - 1][0].text for k in nums]}"
        for label, title, nums in groups)
    logger.info(f"W2.5 chain assignment cell={origin_cell_id} atoms={len(items)} "
                f"new={len(res.new_chains)} appended={sum(res.appended.values())} free={len(res.free)}\n"
                f"  groups: {groups_detail}\n"
                f"  unchained(n={len(res.free)}): {[id2text.get(f, f) for f in res.free]}")
    return res


def _parse_groups(raw_groups, n_atoms: int) -> list[tuple[str, str, list[int]]]:
    """LLM groups -> list of (chain label, title, atom index list); out-of-range indices are dropped
    and a repeated index is ignored on every appearance after the first."""
    out, seen = [], set()
    for g in raw_groups or []:
        if not isinstance(g, dict):
            continue
        label = str(g.get("chain") or "").strip()
        title = str(g.get("title") or "").strip()
        nums = []
        for x in (g.get("atoms") or []):
            try:
                k = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= n_atoms and k not in seen:
                seen.add(k)
                nums.append(k)
        if label and nums:
            out.append((label, title, nums))
    return out


def _execute(store: ChainStore, items: list[tuple[MemoryAtom, np.ndarray]],
             groups: list[tuple[str, str, list[int]]], label2idx: dict[str, int],
             chains: list[tuple[ChainInfo, np.ndarray]],
             res: ChainAssignResult) -> None:
    """Persist: start new chains / append to existing ones. A failing group only loses that group
    (members already appended do no harm) and does not hold up the rest."""
    # Merge existing-chain targets by real chain id first (the LLM may emit two groups pointing at the
    # same chain)
    appends: dict[str, list[int]] = {}
    for label, title, nums in groups:
        if label == "new":
            _exec_new(store, items, nums, title, res)
        elif label in label2idx:
            chain_id = chains[label2idx[label]][0].id
            appends.setdefault(chain_id, []).extend(nums)
        else:
            logger.warning(f"W2.5 ignoring hallucinated chain label {label!r}, its atoms are left unchained")
            res.free.extend(items[k - 1][0].id for k in nums)
    for chain_id, nums in appends.items():
        _exec_append(store, items, chain_id, nums, chains, label2idx, res)
    placed = {k for _, _, nums in groups for k in nums}
    res.free.extend(items[k - 1][0].id for k in range(1, len(items) + 1) if k not in placed)


def _exec_new(store: ChainStore, items: list[tuple[MemoryAtom, np.ndarray]],
              nums: list[int], title: str, res: ChainAssignResult) -> None:
    """Start a new chain: create it from the first member and append the rest in order; the centroid
    is the mean of this group's members."""
    atoms = [items[k - 1] for k in nums]
    first, _ = atoms[0]
    mat = np.stack([v for _, v in atoms])
    info = ChainInfo(title=title or first.text[:40], origin_cell_id=res.origin_cell_id)
    appended = 0
    try:
        store.create_chain(info, first, centroid=mat.mean(axis=0))
        appended = 1   # the first member went in with create_chain
        for a, _v in atoms[1:]:
            store.append_atom(info, a, centroid=None)   # the centroid set at creation already covers every member
            appended += 1
        res.new_chains.append(info)
    except Exception as e:   # noqa: BLE001  failure part-way through: members already appended do no harm, the rest of the group is left free-floating
        logger.warning(f"W2.5 failed to start a new chain, the group's atoms are left unchained: {e}")
        # Only the members not yet appended go back to the free pool — those that made it in must not
        # be reported free, or a later pass could chain them a second time
        res.free.extend(a.id for a, _ in atoms[appended:])


def _exec_append(store: ChainStore, items: list[tuple[MemoryAtom, np.ndarray]],
                 chain_id: str, nums: list[int], chains: list[tuple[ChainInfo, np.ndarray]],
                 label2idx: dict[str, int], res: ChainAssignResult) -> None:
    info = store.get_chain(chain_id)
    if info is None:
        logger.warning(f"W2.5 append target chain does not exist {chain_id}, atoms left unchained")
        res.free.extend(items[k - 1][0].id for k in nums)
        return
    # Group-level incremental centroid: (old centroid x old n + this group's vectors) / (old n + group
    # size), applied once the whole group has been appended
    target = None
    old_pair = next((c for c in chains if c[0].id == chain_id), None)
    if old_pair and old_pair[1] is not None:
        old_n = max(info.n_atoms, 1)
        mat = np.stack([items[k - 1][1] for k in nums])
        target = (np.asarray(old_pair[1], dtype=np.float32) * old_n + mat.sum(axis=0)) / (old_n + len(nums))
    ok = 0
    for k in nums:   # append in within-cell order (chain order = the natural order facts were extracted from the dialogue)
        try:
            store.append_atom(info, items[k - 1][0], centroid=target)
            ok += 1
        except Exception as e:   # noqa: BLE001  one failed member does not hold up the rest of the group
            logger.warning(f"W2.5 append failed atom={items[k - 1][0].id} chain={chain_id}: {e}")
            res.free.append(items[k - 1][0].id)
    if ok:
        res.appended[chain_id] = res.appended.get(chain_id, 0) + ok
