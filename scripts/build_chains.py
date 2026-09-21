"""链回填(docs/atom-chain-design.md §4.3/§8.2):对存量 atom 只重跑判链,不重抽。

场景:开关关闭期写入的 atom 无链,打开开关后跑本脚本补判链;或判链口径升级后重建。

- 幂等:先 chain_store.clear_user()(DELETE 本 user 链行 + atoms 链三列置 NULL,均带
  WHERE;测试护栏拦 TRUNCATE),再按「格时间序 × 格内 occurrence_time 序」重放 W2.5
  (与写入路径同款 assign_chains,一格一次批量 LLM)。
- 评测库(conv-42 等)不走回填走全量重灌:W2 规范收紧改变了抽取本身,旧 atom 没有这条
  地基(§8.2)。本脚本面向 SIT/生产存量。
- 安全:必须显式给 --user / --user-prefix / --all 之一,清链动作带 WHERE 不伤他人。

运行:~/miniconda3/envs/personos/bin/python -m scripts.build_chains --user corpus
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from personos.providers.openai_compat import OpenAIChatLLM, OpenAIEmbedder
from personos.logging_setup import setup_logging
from personos.models import atom_anchor, ensure_aware
from personos.online.chain_build import assign_chains
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.chain_store import ChainStore
from personos.storage.db import Database

_SORT_FLOOR = ensure_aware("1970-01-01")   # 无时间格的排序下界(时间不明按最老重放)


def rebuild_user(db: Database, llm, user_id: str) -> dict:
    """清链重放一个 user 的判链,返回统计。LLM 失败的格按 W2.5 语义留游离(非阻塞)。"""
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
    groups.sort(key=lambda g: g[0])                                   # 格间按格时间序重放

    rebuilt = 0
    for _, mid, items in groups:
        items.sort(key=lambda p: atom_anchor(p[0]) or _SORT_FLOOR)     # 格内按发生时间
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
        "user": user_id or "(默认)",
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
    ap = argparse.ArgumentParser(description="链回填:清本 user 链后按时间序重放 W2.5 判链")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--user", help="精确 user_id(共享库按命名空间隔离)")
    g.add_argument("--user-prefix", help="user_id 前缀(批量回填,如 locomo-)")
    g.add_argument("--all", action="store_true", help="atoms 表里全部 user(慎用)")
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
        print(f"[{s['user']}] 清链 {s['cleared_chains']} → 重建分配 {s['rebuilt_assigned']} atoms "
              f"/ {s['n_cells_replayed']} 格 | atoms={s['n_atoms']} chains={s['n_chains']} "
              f"chained={s['chained_atoms']} 游离率={s['free_rate']}")
        for n, title in s["top_chains"]:
            print(f"    · «{title}» n={n}")


if __name__ == "__main__":
    main()
