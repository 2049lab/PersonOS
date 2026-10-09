"""Stage 0: record the **behavioural baseline** for the open-sourcing effort.
It produces three portable artefacts:

  baseline/corpus_snapshot.json  a full-table snapshot after loading a fixed
                                 corpus (no LLM involved; can be INSERTed
                                 straight into any backend)
  baseline/cassette.json         every model response seen during recall
                                 (keyed by prompt hash), for offline replay
  baseline/recall_golden.json    the structural RecallOutcome fingerprint for
                                 each probe question

Why all three are indispensable: the snapshot pins down "what the memory store
looks like" and the cassette pins down "what the model said". Together they
turn recall into a **purely deterministic function**, which is the only reason
the fingerprints are comparable at all. Drop any one of them and a difference
in the comparison could just as easily be environmental noise as a real change.

Usage:
  record  PYTHONPATH=. python -m scripts.baseline.record --record
  verify  PYTHONPATH=. python -m scripts.baseline.record --verify   (replay and compare)

--record really calls the LLM while loading the corpus (about 29 sessions /
186 turns); it is slow and consumes quota, so run it once before the work
starts. --verify is entirely offline, can be run at any time, and should be run
at the end of every stage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from personos.models import ensure_aware, now
from scripts.baseline.cassette import read_json, write_json

BASE = Path("baseline")
SNAP = BASE / "corpus_snapshot.json.gz"      # mostly 4096-dim vectors; compresses to about a fifth
TAPE = BASE / "cassette.json.gz"
GOLD = BASE / "recall_golden.json"           # small and meant to be read by humans, so uncompressed
USER = "golden_baseline"
SESSION = "golden"
# **"Now" must be frozen.** The prompts for the R0 rewrite, for answering and
# for review all embed the current time. With now(), every run would produce a
# different prompt, the hash would never match, and the cassette could never be
# replayed. This was learned the hard way; do not change it back.
FROZEN_NOW = "2026-09-21T12:00:00+08:00"
# Tables covered by the snapshot. The three identity tables are excluded (a
# plain-text corpus produces no character rows).
# The value is a stable sort key: the snapshot needs a deterministic order, or
# two exports of the same data would diff against each other for no reason.
# session_context has no id column (its primary key is user_id + session_id),
# which is what made an earlier version of this query fail.
TABLES = {"evidence": "id", "memcells": "id", "atoms": "id", "atom_chains": "id",
          "session_context": "session_id"}


def _date(s: str) -> datetime:
    try:
        return ensure_aware(datetime.fromisoformat(str(s)[:10]))
    except Exception:
        return now()


def _probes(corpus: dict) -> list[dict]:
    return corpus.get("probes", [])


# -- Record ---------------------------------------------------------------

def record(corpus_path: str, *, skip_ingest: bool = False) -> int:
    from personos.online.write_path import SessionWriter
    from personos.providers.openai_compat import OpenAIChatLLM, OpenAIEmbedder
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.chain_store import ChainStore
    from personos.storage.db import Database
    from personos.storage.evidence_store import EvidenceStore
    from scripts.baseline.cassette import Cassette, RecordingEmbedder, RecordingLLM
    from scripts.baseline.golden import fingerprint

    corpus = json.load(open(corpus_path, encoding="utf-8"))
    sessions = [(s.get("session_id", ""), _date(s.get("date", "")), s.get("turns", []))
                for p in corpus.get("periods", []) for s in p.get("sessions", [])]
    sessions.sort(key=lambda x: (x[1], x[0]))

    db = Database()
    if not skip_ingest:   # idempotent re-run: clear this user only (a table-wide DELETE would hit others)
        for t in TABLES:
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (USER,))
    ev, at = EvidenceStore(db, USER), AtomStore(db, USER)
    cells, chains = CellStore(db, USER), ChainStore(db, USER)
    llm = OpenAIChatLLM(timeout=120.0)
    embedder = OpenAIEmbedder(timeout=120.0)

    # Loading the corpus really does take ~50 minutes over 29 sessions, so a
    # failure in the later export/probe stages must not force it to be redone.
    print("1. loading corpus: " + ("skipped (reusing what is already in the database)" if skip_ingest
                                   else f"{len(sessions)} sessions -> user={USER} (real LLM, be patient)"),
          flush=True)
    for i, (sid, dt, turns) in enumerate([] if skip_ingest else sessions, 1):
        w = SessionWriter(llm, embedder, ev, cells, at, session_id=f"{SESSION}-{sid}",
                          user_id=USER, chain_store=chains)
        n = 0
        for t in turns:
            for role, key in (("user", "user"), ("assistant", "assistant")):
                msg = (t.get(key) or "").strip()
                if msg:
                    w.feed(role, msg, now_dt=dt, source_extra={"system": "golden"})
                    n += 1
        w.end_session()
        print(f"   [{i}/{len(sessions)}] {sid} @ {dt.date()} {n} messages", flush=True)

    print("\n2. exporting the full-table snapshot", flush=True)
    snap = {t: db.fetch_all(f"SELECT * FROM {t} WHERE user_id=%s ORDER BY {k}", (USER,))
            for t, k in TABLES.items()}
    snap = {t: [{k: _jsonable(v) for k, v in row.items()} for row in rows]
            for t, rows in snap.items()}
    BASE.mkdir(exist_ok=True)
    write_json(SNAP, snap)
    print("   " + "  ".join(f"{t}={len(rows)}" for t, rows in snap.items()))

    print("\n3. running the probes and recording the cassette", flush=True)
    tape = Cassette()
    rllm, remb = RecordingLLM(llm, tape), RecordingEmbedder(embedder, tape)
    golden = {}
    for pr in _probes(corpus):
        q = pr.get("question", "")
        o = _run(rllm, remb, ev, cells, at, q)
        golden[q] = fingerprint(o)
        print(f"   ok [{pr.get('kind','?')}] {q[:38]}  "
              f"ranked={len(o.ranked)} verdicts={[r.verdict for r in o.reviews]} "
              f"escalated={o.escalated}", flush=True)

    tape.save(TAPE)
    GOLD.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ncassette {tape.stats()}\nbaseline written: {SNAP} / {TAPE} / {GOLD}")
    db.close()
    return 0


def _jsonable(v):
    import datetime as _dt
    if isinstance(v, (bytes, bytearray)):
        return {"__bytes_hex__": v.hex()}
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return v


def restore(db) -> dict[str, int]:
    """Load the snapshot back into whichever database is configured, clearing
    this user first. **Verification has to carry its own data.** Otherwise the
    baseline is tied to leftover rows in one particular database: it stops
    working the moment somebody clears that database, and more importantly
    there is nothing to compare against once the backend changes. Carrying the
    data is what makes this baseline portable."""
    snap = read_json(SNAP)
    for t in TABLES:
        db.execute(f"DELETE FROM {t} WHERE user_id=%s", (USER,))
    counts = {}
    for t in TABLES:
        rows = snap.get(t, [])
        counts[t] = len(rows)
        for row in rows:
            cols, ph, vals = [], [], []
            for c, v in row.items():
                cols.append(c)
                # BLOBs must go through the hex text channel: some database
                # proxies are not binary-safe for pymysql's _binary'...'
                # literals (high bytes and NUL trigger errors). See
                # storage/db.blob_param; this reuses the same convention rather
                # than inventing a second one.
                if isinstance(v, dict) and "__bytes_hex__" in v:
                    ph.append("UNHEX(%s)")
                    vals.append(v["__bytes_hex__"])
                else:
                    ph.append("%s")
                    vals.append(v)
            db.execute(f"INSERT INTO {t} ({','.join(cols)}) VALUES ({','.join(ph)})",
                       tuple(vals))
    return counts


def _run(llm, emb, ev, cells, at, query: str):
    """Run one fast-path recall. **The deep path is off**: it is an LLM agent
    whose step count and tool choices are extremely sensitive to temperature,
    so putting it in the baseline would only manufacture false alarms. Its
    structural changes are guarded by test_deep_recall instead."""
    from personos.online.recall_flow import run_recall
    return run_recall(llm, emb, at, cells, ev, session_id=SESSION, query=query,
                      now_dt=ensure_aware(datetime.fromisoformat(FROZEN_NOW)),
                      mode="fast", top_k=30)


# -- Verify ---------------------------------------------------------------

def verify() -> int:
    from personos.storage.atom_store import AtomStore
    from personos.storage.cell_store import CellStore
    from personos.storage.db import Database
    from personos.storage.evidence_store import EvidenceStore
    from scripts.baseline.cassette import Cassette, CassetteMiss, ReplayEmbedder, ReplayLLM
    from scripts.baseline.golden import diff, fingerprint

    for p in (SNAP, TAPE, GOLD):
        if not p.exists():
            print(f"missing baseline artefact {p} - run --record first")
            return 2

    tape = Cassette.load(TAPE)
    llm, emb = ReplayLLM(tape), ReplayEmbedder(tape)
    golden = json.loads(GOLD.read_text(encoding="utf-8"))

    db = Database()
    ok, n_bad = True, 0
    try:
        print(f"1. restoring the snapshot -> {restore(db)}", flush=True)
        ev, at, cells = EvidenceStore(db, USER), AtomStore(db, USER), CellStore(db, USER)
        print("2. replaying the probes offline", flush=True)
        for q, want in golden.items():
            try:
                got = fingerprint(_run(llm, emb, ev, cells, at, q))
            except CassetteMiss as e:  # a miss means the prompt changed: the most valuable failure
                print(f"FAIL {q[:40]}\n   {e}")
                ok, n_bad = False, n_bad + 1
                continue
            d = diff(want, got)
            if d:
                ok, n_bad = False, n_bad + 1
                print(f"FAIL {q[:40]}")
                for line in d:
                    print(f"   - {line}")
            else:
                print(f"ok   {q[:40]}")
    finally:
        # Leave no residue in a shared database (a project convention): the
        # snapshot carries its own data, so the copy in the database is deleted
        # as soon as it has been used. This sits in finally so that a mid-run
        # failure still cleans up.
        for t in TABLES:
            try:
                db.execute(f"DELETE FROM {t} WHERE user_id=%s", (USER,))
            except Exception:  # noqa: BLE001
                pass
        db.close()
    print(f"\nBASELINE_VERIFY {'PASS' if ok else f'FAIL ({n_bad}/{len(golden)} fingerprints differ)'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="run for real and record the baseline (slow, consumes quota)")
    ap.add_argument("--verify", action="store_true",
                    help="replay offline and compare against the baseline")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse the corpus already in the database and redo only the snapshot "
                         "export and the probes (loading is expensive, do not waste it)")
    ap.add_argument("--corpus", default="data/persona_gen/corpus.json")
    a = ap.parse_args()
    if a.record:
        return record(a.corpus, skip_ingest=a.skip_ingest)
    if a.verify:
        return verify()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
