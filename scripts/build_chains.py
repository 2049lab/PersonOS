"""Chain backfill (docs/atom-chain-design.md 4.3/8.2): re-run chain assignment
over existing atoms without re-extracting them.

When it applies: atoms written while the feature flag was off carry no chain,
so run this after turning the flag on; or run it to rebuild after the chain
criteria have been revised.

- Idempotent: first `chain_store.clear_user()` (DELETE this user's chain rows
  and NULL out the three chain columns on their atoms, both with a WHERE
  clause; a test guard rejects TRUNCATE), then replay W2.5 in "cell time order
  x per-cell occurrence_time order" using the same `assign_chains` the write
  path uses (one batched LLM call per cell).
- Benchmark datasets (conv-42 and friends) are reloaded from scratch rather
  than backfilled: the tightened W2 specification changed extraction itself,
  and old atoms do not rest on that foundation (8.2). This script targets
  production-style existing data.
- Safety: exactly one of --user / --user-prefix / --all is required, and the
  chain-clearing statements always carry a WHERE clause so other users are
  untouched.

Usage: python -m scripts.build_chains --user corpus
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from personos.providers.openai_compat import OpenAIChatLLM
from personos.logging_setup import setup_logging
from personos.models import atom_anchor, ensure_aware
from personos.online.chain_build import assign_chains
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.db import Database

_SORT_FLOOR = ensure_aware("1970-01-01")   # sort floor for undated cells (replay them first)


def rebuild_user(db: Database, llm, user_id: str) -> dict:
    """Clear and replay chain assignment for one user; return statistics.

    Cells whose LLM call fails are left free-floating, per W2.5 semantics
    (non-blocking).
    """
    chains = ChainStore(db, user_id)
    cleared = chains.clear_user()
    atoms = AtomStore(db, user_id)
    cells = CellStore(db, user_id)

    by_cell: dict[str, list] = defaultdict(list)
    for a, vec in atoms.all_with_embeddings():
        if a.memcell_id:
            by_cell[a.memcell_id].append((a, vec))
    groups = []
    for mid, items in by_cell.items():
        c = cells.get(mid)
        groups.append(((ensure_aware(c.t_start) if c else None) or _SORT_FLOOR, mid, items))
    groups.sort(key=lambda g: g[0])                                   # replay cells in time order

    rebuilt = 0
    for _, mid, items in groups:
        items.sort(key=lambda p: atom_anchor(p[0]) or _SORT_FLOOR)     # within a cell, by occurrence
        res = assign_chains(llm, chains, items, origin_cell_id=mid)
        rebuilt += res.assigned
    return _stats(db, user_id, cleared_chains=cleared, rebuilt=rebuilt, n_cells=len(groups))


def _stats(db: Database, user_id: str, *, cleared_chains: int, rebuilt: int, n_cells: int) -> dict:
    chains = ChainStore(db, user_id)
    atoms = AtomStore(db, user_id)
    infos = [info for info, _ in chains.list_chains()]
    rows = atoms.all_with_embeddings()
    chained = sum(1 for a, _ in rows if a.chain_id)
    stats = {
        "user": user_id or "(default)",
        "cleared_chains": cleared_chains,
        "rebuilt_assigned": rebuilt,
        "n_cells_replayed": n_cells,
        "n_atoms": len(rows),
        "n_chains": len(infos),
        "chained_atoms": chained,
        "free_rate": round(1 - chained / len(rows), 3) if rows else 0.0,
        "top_chains": sorted(((c.n_atoms, c.title) for c in infos), reverse=True)[:10],
    }
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Chain backfill: clear this user's chains, then replay W2.5 in time order")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", help="exact user_id (users are namespaced in a shared database)")
    g.add_argument("--user-prefix", help="user_id prefix (bulk backfill, e.g. locomo-)")
    g.add_argument("--all", action="store_true", help="every user in the atoms table (use with care)")
    args = ap.parse_args()

    setup_logging(Path("logs"))
    db = Database()
    if args.user:
        users = [args.user]
    else:
        rows = db.fetch_all("SELECT DISTINCT user_id FROM atoms")
        users = sorted(r["user_id"] for r in rows)
        if args.user_prefix:
            users = [u for u in users if u.startswith(args.user_prefix)]

    llm = OpenAIChatLLM(timeout=120.0)
    for u in users:
        s = rebuild_user(db, llm, u)
        print(f"[{s['user']}] cleared {s['cleared_chains']} chains -> reassigned "
              f"{s['rebuilt_assigned']} atoms across {s['n_cells_replayed']} cells | "
              f"atoms={s['n_atoms']} chains={s['n_chains']} chained={s['chained_atoms']} "
              f"free_rate={s['free_rate']}")
        for n, title in s["top_chains"]:
            print(f"    - <{title}> n={n}")


if __name__ == "__main__":
    main()
