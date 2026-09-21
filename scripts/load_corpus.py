"""把 generate_corpus.py 造的 corpus.json 按时间顺序逐会话灌进 PersonOS(真实写入链路)。

- 全部会话按 date 升序灌(时序正确:先发生的先入库,演化/纠正在作答时按 cell 时间窗消费)。
- 每个会话一个 SessionWriter:逐轮 feed(user 句 + assistant 句,backdate 到会话日期),
  会话末 end_session 强制闭合(分段覆盖全会)。
- 灌完打印 corpus 里的 probes(可直接抄进工作台提问)。

运行:~/miniconda3/envs/personos/bin/python -m scripts.load_corpus --corpus data/persona_gen/corpus.json [--user-id corpus]
然后:python run_dev.py → 工作台选中对应 user 提问(共享 SIT 库,user 命名空间隔离)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from personos.clients.maas import MaasClient
from personos.logging_setup import setup_logging
from personos.models import ensure_aware, now
from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from personos.storage.user_store import UserStore


def _parse_date(s: str) -> datetime:
    try:
        return ensure_aware(datetime.fromisoformat(str(s)[:10]))
    except Exception:
        return now()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/persona_gen/corpus.json")
    ap.add_argument("--user-id", default="corpus", help="user 命名空间(重跑清本 user 四表)")
    args = ap.parse_args()

    setup_logging(Path("logs"))
    corpus = json.load(open(args.corpus, encoding="utf-8"))

    # 展平所有会话,按日期升序(跨段时序正确:先发生的先入库)
    sessions = []
    for p in corpus.get("periods", []):
        for s in p.get("sessions", []):
            sessions.append((s.get("session_id", ""), _parse_date(s.get("date", "")), s.get("turns", [])))
    sessions.sort(key=lambda x: (x[1], x[0]))

    db = Database()
    # 幂等:清本 user 的四表(全表 DELETE 会伤其他 user,严禁)
    for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
        db.execute(f"DELETE FROM {t} WHERE user_id=%s", (args.user_id,))
    ev, at, cells = EvidenceStore(db, args.user_id), AtomStore(db, args.user_id), \
        CellStore(db, args.user_id)
    maas = MaasClient(timeout=120.0)

    total_turns = 0
    for sid, dt, turns in sessions:
        writer = SessionWriter(maas, maas, ev, cells, at, session_id=f"corpus-{sid}")
        fed = 0
        for t in turns:
            msg = (t.get("user") or "").strip()
            if msg:
                writer.feed("user", msg, now_dt=dt, source_extra={"system": "corpus"})
                fed += 1
            reply = (t.get("assistant") or "").strip()
            if reply:
                writer.feed("assistant", reply, now_dt=dt, source_extra={"system": "corpus"})
                fed += 1
        total_turns += fed
        writer.end_session()   # 会话末强制闭合收尾段(分段覆盖全会)
        print(f"  {sid} @ {dt.date()}  {fed:>3} 句 → cells 累计 {len(writer.cells)}")

    try:
        cred = UserStore(db).register(args.user_id)
        print(f"\n注册 user「{args.user_id}」成功,token(请保存,只显示一次): {cred['token']}")
    except ValueError:
        print(f"\nuser「{args.user_id}」已存在,沿用旧 token")
    print(f"落库完成 → MySQL user 命名空间 {args.user_id!r}")
    print(f"  会话 {len(sessions)} | 灌入 {total_turns} 句 | "
          f"cells {len(cells.iter_all())} | atoms {len(at.list(limit=100000))} | "
          f"证据 {len(ev.list(limit=100000))}")
    print("\n可直接抄的提问(corpus 自带 probes,带期望答案便于你对答案):")
    for pr in corpus.get("probes", []):
        print(f"  • [{pr.get('kind','?')}] {pr.get('question','')}")
        print(f"        期望: {pr.get('expected','')}")
    print(f"\n开工作台提问:")
    print(f"  ~/miniconda3/envs/personos/bin/python run_dev.py  → http://127.0.0.1:8000/")
    db.close()


if __name__ == "__main__":
    main()
