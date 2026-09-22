"""Load the corpus.json produced by generate_corpus.py into PersonOS, session by
session in chronological order, through the real write path.

- Every session is loaded in ascending date order, so that what happened first
  is stored first; evolutions and corrections are then consumed at answer time
  according to each cell's time window.
- One SessionWriter per session: turns are fed one at a time (the user line and
  the assistant line, backdated to the session date), and end_session() forces
  the trailing segment closed so that segmentation covers the whole session.
- After loading, the probes carried inside the corpus are printed so they can be
  pasted straight into the console.

Run:  python -m scripts.load_corpus --corpus data/persona_gen/corpus.json [--user-id corpus]
Then: start the API server (uvicorn server.app:app --reload) and ask questions as
the matching user (users are namespaced, so a shared database stays isolated per user).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from personos.providers.openai_compat import OpenAIChatLLM, OpenAIEmbedder
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
    ap.add_argument("--user-id", default="corpus",
                    help="user namespace (a re-run clears this user's rows in the business tables)")
    args = ap.parse_args()

    setup_logging(Path("logs"))
    corpus = json.load(open(args.corpus, encoding="utf-8"))

    # Flatten every session and sort by date, so ordering stays correct across
    # periods: what happened first is stored first.
    sessions = []
    for p in corpus.get("periods", []):
        for s in p.get("sessions", []):
            sessions.append((s.get("session_id", ""), _parse_date(s.get("date", "")), s.get("turns", [])))
    sessions.sort(key=lambda x: (x[1], x[0]))

    db = Database()
    # Idempotent: clear this user's rows only. A table-wide DELETE would
    # destroy other users' data and is forbidden.
    for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
        db.execute(f"DELETE FROM {t} WHERE user_id=%s", (args.user_id,))
    ev, at, cells = EvidenceStore(db, args.user_id), AtomStore(db, args.user_id), \
        CellStore(db, args.user_id)
    llm = OpenAIChatLLM(timeout=120.0)
    embedder = OpenAIEmbedder(timeout=120.0)

    total_turns = 0
    for sid, dt, turns in sessions:
        writer = SessionWriter(llm, embedder, ev, cells, at, session_id=f"corpus-{sid}")
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
        writer.end_session()   # force the trailing segment closed at end of session
        print(f"  {sid} @ {dt.date()}  {fed:>3} messages -> {len(writer.cells)} cells so far")

    try:
        cred = UserStore(db).register(args.user_id)
        print(f"\nregistered user {args.user_id!r}; token (save it, shown only once): {cred['token']}")
    except ValueError:
        print(f"\nuser {args.user_id!r} already exists, reusing the existing token")
    print(f"load complete -> user namespace {args.user_id!r}")
    print(f"  sessions {len(sessions)} | messages {total_turns} | "
          f"cells {len(cells.iter_all())} | atoms {len(at.list(limit=100000))} | "
          f"evidence {len(ev.list(limit=100000))}")
    print("\nReady-made questions (the corpus ships probes with expected answers):")
    for pr in corpus.get("probes", []):
        print(f"  - [{pr.get('kind','?')}] {pr.get('question','')}")
        print(f"        expected: {pr.get('expected','')}")
    print("\nStart the console:")
    print("  uvicorn server.app:app --reload   ->  http://127.0.0.1:8000/api-doc")
    db.close()


if __name__ == "__main__":
    main()
