"""LongMemEval x personos benchmark harness, one question at a time: each
instance carries its own history.

Flow: pick a question -> load its sessions into a fresh store one by one
(SessionWriter, with now_dt set to each session's timestamp) -> run the whole
pipeline through run_recall as of question_date -> answer with answer_mode_a ->
score with the judge, under both conventions (the Mem0 convention judges the
answerer's output, the product convention judge_r5 judges the R5 answer
directly).
The gold evidence chain is located at session level (answer_session_ids) plus
the has_answer turns, and the break-point taxonomy matches run_locomo
(evidence -> extraction -> retrieval -> materials -> adoption -> answering).

Differences from run_locomo: each question gets its own store (haystacks are
never shared), the dialogue is user-assistant, and the gold answer of a
counting question is normalized with str() before judging.

Usage (single question):
  python -m scripts.bench.run_longmemeval --idx 0 --out data/longmemeval/runs/q0
  python -m scripts.bench.run_longmemeval --qid <question_id> ...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

from personos.online.recall_flow import run_recall
from personos.providers.openai_compat import OpenAIEmbedder, OpenAIReranker
from personos.providers.anthropic_compat import AnthropicChatLLM
from personos.online.rerank import ScoringReranker
from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from scripts.bench.judge import Verdict, answer_mode_a, judge
from scripts.bench.longmemeval_adapter import LmeQuestion, load_questions
from scripts.bench.run_locomo import _ANSWER_MEM_CELLS, _atom_view, _cap, _grouped_materials, _hit_view

LME_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")


def ingest(q: LmeQuestion, llm, embedder, ev_store, cells, atoms) -> tuple[list, dict]:
    """Load the store session by session; returns (per-session summaries,
    turn -> evidence_id mapping)."""
    summaries, turn2ev = [], {}
    for s in q.sessions:
        writer = SessionWriter(llm, embedder, ev_store, cells, atoms,
                               session_id=f"{q.qid}-s{s.idx}")
        t0, closed = time.time(), []
        for t in s.turns:
            r = writer.feed(t.holder, t.text, now_dt=s.dt)
            turn2ev[f"{s.idx}:{t.turn_idx}"] = r.evidence_id
            if r.closed_cell:
                closed.append(r.closed_cell)
        closed.extend(writer.end_session()[len(closed):])
        summaries.append({"session": s.idx, "sid": s.sid, "dt": s.dt.isoformat(),
                          "turns": len(s.turns), "cells": len(closed),
                          "secs": round(time.time() - t0, 1)})
        logger.info(f"[{q.qid}] s{s.idx}({s.sid[:20]}) {len(s.turns)} turns -> "
                    f"{len(closed)} cells ({summaries[-1]['secs']}s)")
    return summaries, turn2ev


def gold_chain(q: LmeQuestion, turn2ev: dict, store_atoms: list, qa_rec: dict) -> list:
    """Evidence chain: for each evidence session, follow its has_answer turns to
    their evidence ids and on through extraction, retrieval, materials and
    citation, recording how far each got."""
    ev2atoms: dict[str, list] = defaultdict(list)
    for a in store_atoms:
        for e in a.get("evidence") or []:
            ev2atoms[e].append(a)
    hit_ids = {x["id"] for h in qa_rec.get("hits", []) for x in h["atoms"]}
    mat_ids = {x["id"] for h in qa_rec.get("ranked", []) for x in h["atoms"]}
    cited = set((qa_rec.get("r5") or {}).get("cited_cells") or [])
    chains = []
    for sidx in q.evidence_sidx:
        s = q.sessions[sidx]
        evs = [turn2ev.get(f"{sidx}:{t.turn_idx}") for t in s.turns if t.has_answer]
        hit_atoms = [a for e in filter(None, evs) for a in ev2atoms.get(e, [])]
        chains.append({
            "session": sidx, "sid": s.sid, "n_answer_turns": len(evs),
            "n_extracted": len(hit_atoms),
            "extracted_atoms": [a["text"][:60] for a in hit_atoms[:3]],
            "in_hits": any(a["id"] in hit_ids for a in hit_atoms),
            "in_material": any(a["id"] in mat_ids for a in hit_atoms),
            "cited": bool(hit_atoms and (cited & {a["cell_id"] for a in hit_atoms if a.get("cell_id")})),
        })
    return chains


def break_point(qa_rec: dict, chains: list) -> str:
    if not chains:
        return "no evidence"
    if not any(c["n_extracted"] for c in chains):
        return "extraction (the verbatim text is stored but no atom was extracted)"
    if any(c["cited"] for c in chains):
        ans = qa_rec.get("answer") or ""
        if re.match(r"i don'?t (have|know)|no record|not (explicitly|mentioned)", ans.lower()):
            return "answerer (refused despite citing the evidence)"
        return "answering (the answerer drifted or the judge misjudged; review by hand)"
    if any(c["in_material"] for c in chains):
        return "adoption (reached the answering materials but R5 did not cite it)"
    if any(c["in_hits"] for c in chains):
        return "materials (retrieved but cut off by the top-20 limit)"
    return "retrieval (the atom is stored but neither path recalled it)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--qid", default="", help="select by question_id (overrides --idx)")
    ap.add_argument("--data", default=str(LME_PATH))
    ap.add_argument("--out", default="data/longmemeval/runs/q0")
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "deep"])
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help=">0 loads only the first N sessions (for debugging)")
    args = ap.parse_args()

    qs = load_questions(args.data)
    q = next(x for x in qs if x.qid == args.qid) if args.qid else qs[args.idx]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.limit_sessions:
        q.sessions = q.sessions[:args.limit_sessions]
        # Drop evidence session indices that the truncation put out of range, so
        # gold_chain cannot go out of bounds. Smoke/debug use only.
        q.evidence_sidx = [i for i in q.evidence_sidx if i < len(q.sessions)]
    print(f"question {q.qid} [{q.qtype}{'+abs' if q.abstention else ''}]: {q.question}")
    print(f"  gold={str(q.answer)[:80]} | {len(q.sessions)} sessions / {q.n_turns()} turns / "
          f"evidence sessions {q.evidence_sidx} | asked as of {q.question_dt.isoformat()[:16]}")

    # Each question gets its own user namespace, and a re-run clears that user's
    # four tables first (idempotent, so atom counts do not double).
    uid = f"lme-{q.qid}"
    db = Database()
    for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
        db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
    ev_store, atoms, cells = EvidenceStore(db, uid), AtomStore(db, uid), CellStore(db, uid)
    llm, embedder, scorer = AnthropicChatLLM(), OpenAIEmbedder(), OpenAIReranker()
    t0 = time.time()
    sessions_summary, turn2ev = ingest(q, llm, embedder, ev_store, cells, atoms)
    n_atoms = len(atoms.list(limit=100000))
    print(f"load complete ({time.time()-t0:.0f}s): {len(q.sessions)} sessions -> "
          f"{len(list(cells.iter_all(limit=10000)))} cells / {n_atoms} atoms")

    t1 = time.time()
    rec = None
    for attempt in range(1, 4):
        try:
            o = run_recall(llm, embedder, atoms, cells, ev_store,
                           session_id=f"{q.qid}-qa", query=q.question,
                           now_dt=q.question_dt, mode=args.mode, reranker=ScoringReranker(scorer))
            if o.deep and not o.ranked:
                groups = [(c, atoms.list_by_cell(c.id))
                          for c in (cells.get(cid) for cid in (o.ans.cited_cells if o.ans else [])) if c]
            else:
                groups = [(h.cell, [sa.atom for sa in h.atoms]) for h in o.ranked[:_ANSWER_MEM_CELLS]]
            answer = answer_mode_a(llm, question=q.question, brief=o.ans.answer,
                                   mem_block=_grouped_materials(groups))
            v = judge(llm, question=q.question, gold=str(q.answer), prediction=answer)
            try:   # the product convention judges R5 directly; a scoring failure is recorded, not retried
                v5 = judge(llm, question=q.question, gold=str(q.answer),
                           prediction=o.ans.answer if o.ans else "")
            except Exception:   # noqa: BLE001
                logger.exception("the product-convention judge failed")
                v5 = Verdict(ok=False, raw="ERROR: judge_r5 failed (see the run log)")
            rec = {
                "question_type": q.qtype, "abstention": q.abstention,
                "question": q.question, "gold": q.answer, "answer": answer,
                "judge": v.ok, "judge_raw": v.raw,
                "judge_r5": v5.ok, "judge_r5_raw": v5.raw,
                "rewrite": {"resolved": o.rw.resolved if o.rw else None,
                            "expansions": o.rw.expansions if o.rw else []},
                "pool": [_atom_view(a, i + 1) for i, a in enumerate(o.hits[:10])],
                "hits": [_hit_view(h, i + 1) for i, h in
                         enumerate((o.asm.units if o.asm else [])[:10])],
                "ranked": [_hit_view(h, i + 1) for i, h in enumerate(o.ranked[:10])],
                "review": ({"verdict": o.reviews[-1].verdict, "critique": o.reviews[-1].critique,
                            "retried": o.retried} if o.reviews else None),
                "r5": {"answer": o.ans.answer, "cited_cells": o.ans.cited_cells} if o.ans else None,
                "secs": round(time.time() - t1, 1),
            }
            break
        except Exception:   # noqa: BLE001
            logger.exception(f"QA failed, attempt {attempt}/3")
            if attempt < 3:
                time.sleep(60)
    if rec is None:
        rec = {"question": q.question, "gold": q.answer, "answer": "", "judge": False,
               "judge_raw": "ERROR: still failing after 3 retries", "judge_r5": False,
               "judge_r5_raw": "ERROR: still failing after 3 retries", "secs": round(time.time() - t1, 1)}

    store_atoms = [{"id": a.id, "cell_id": a.memcell_id, "text": a.text,
                    "evidence": [r.evidence_id for r in a.evidence_refs]}
                   for a in atoms.list(limit=100000)]
    rec["gold_chain"] = gold_chain(q, turn2ev, store_atoms, rec)
    rec["break_point"] = break_point(rec, rec["gold_chain"])

    trace = {"meta": {"qid": q.qid, "qtype": q.qtype, "abstention": q.abstention,
                      "n_sessions": len(q.sessions), "n_turns": q.n_turns(),
                      "mode": args.mode, "llm": "MiniMax-M3", "data": args.data,
                      "run_at": time.strftime("%Y-%m-%d %H:%M"),
                      **({"git": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                                capture_output=True, text=True).stdout.strip()}
                         if not subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).returncode else {})}}
    try:
        trace["meta"]["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                              capture_output=True, text=True).stdout.strip()
    except Exception:   # noqa: BLE001
        pass
    trace["ingest"] = sessions_summary
    trace["qa"] = rec
    trace["store"] = {"n_cells": len(list(cells.iter_all(limit=10000))), "n_atoms": len(store_atoms)}
    (out.with_suffix(".trace.json")).write_text(json.dumps(trace, ensure_ascii=False, indent=1))

    mark = "ok" if rec["judge"] else "X"
    print(f"\n===== {mark} [{rec.get('question_type')}] ({rec['secs']}s, "
          f"bp={rec['break_point'][:20]}) =====")
    print(f"  gold:   {str(q.answer)[:150]}")
    print(f"  answer: {(rec['answer'] or '')[:200]}")
    print(f"  judge_raw: {str(rec['judge_raw'])[:80]}")
    print(f"trace: {out.with_suffix('.trace.json')}")
    embedder.close(); scorer.close(); llm.close(); db.close()


if __name__ == "__main__":
    main()
