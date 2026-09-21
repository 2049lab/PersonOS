"""LoCoMo x personos benchmark harness, with a full-pipeline trace written to disk.

Flow: take the first N sessions of a conversation -> feed them message by
message through SessionWriter (W0 evidence / W1 boundaries / W2 cell building,
with now_dt set to the session time) -> select the questions whose evidence
lies entirely within the loaded sessions -> run the whole fast path in
run_recall (R0 scope -> R1 dual retrieval -> R2 reranking -> R5 draft ->
R3' review) -> produce an English answer with the mode-A answerer -> score it
with a Mem0-style binary judge.

The benchmark protocol reports two scoring conventions:
- the **Mem0 convention** (comparable with published numbers): the answerer is
  mode A (brief = the R5 answer; memories = the atoms hit in the top 20
  reranked cells, dated and grouped by cell, aligned with what R5 saw, under
  the trust-the-brief rule), and the judge scores the answerer's output;
- the **product convention** (what the product actually returns): no second
  stage, and the judge scores the R5 answer directly (judge_r5).
State the protocol difference whenever these numbers are compared externally.
Score differences are attributed to changes in the memory system itself
(writing, retrieval, and how R5 organizes materials).

Transparency is a first-class concern here: the boundary decision for every
turn, the topic/episode/atoms of every closed cell, and the scope, dual-path
ranking, review and answer for every question all go into the trace JSON, which
report.py renders into a self-contained HTML report.

Usage:
  python -m scripts.bench.run_locomo --conv conv-26 --n-sessions 99 --n-questions 999
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from personos.online.recall_flow import run_recall
from personos.providers.openai_compat import OpenAIEmbedder, OpenAIReranker
from personos.providers.anthropic_compat import AnthropicChatLLM
from personos.models import stamped_atom_text
from personos.online.rerank import ScoringReranker
from personos.online.retrieval import cell_lead
from personos.online.trust import evidence_entries
from personos.online.write_path import SessionWriter
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from scripts.bench.judge import Verdict, answer_mode_a, judge
from scripts.bench.locomo_adapter import LocomoConversation, load_conversations, pick_answerable

LOCOMO_PATH = Path("data/locomo10.json")   # local by default; override with --data elsewhere

# Truncation limit for long text in the trace (these are what the report lets
# you expand; the complete verbatim text lives in the store and is never cut).
_CAP_RAW = 2000

# How wide the answerer's memory view is: the atoms hit in the top N reranked
# cells, the same convention the public /recall uses for supporting memories.
_ANSWER_MEM_CELLS = 20    # aligned with what R5 sees (10 made the two layers disagree)


def _grouped_materials(groups: list[tuple]) -> str:
    """Organize the answerer's materials: group by cell, each group headed by
    cell_lead (dialogue time plus topic) and followed by its stamped atoms.

    Structurally identical to the material shape answer_from_cells uses in the
    product: list questions are re-checked group by group, and the group header
    supplies the time and topic anchor.
    """
    parts = []
    for c, atom_list in groups:
        lines = "\n".join(f"- {stamped_atom_text(a)}" for a in atom_list)
        parts.append(f"{cell_lead(c)}\n{lines}" if lines else cell_lead(c))
    return "\n\n".join(parts)


def _cap(s: str | None, n: int = _CAP_RAW) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" ...[truncated; full text is {len(s)} chars]"


def _atom_brief(a) -> dict:
    return {
        "id": a.id, "cell_id": a.memcell_id, "holder": a.holder, "text": a.text,
        "type": a.object_type, "domains": a.domains, "kind": a.kind,
        "occurrence": a.occurrence_time.isoformat() if a.occurrence_time else None,
        "evidence": [r.evidence_id for r in a.evidence_refs],
    }


def _cell_brief(cb) -> dict:
    """Trace view of a closed cell: topic/episode/atoms plus the provenance of
    the two W2 calls."""
    c = cb.cell
    return {
        "cell_id": c.id, "topic": c.topic, "domains": c.domains,
        "t_start": c.t_start.isoformat() if c.t_start else None,
        "t_end": c.t_end.isoformat() if c.t_end else None,
        "episode": _cap(c.episode, 600),
        "atoms": [_atom_brief(a) for a in cb.atoms],
        "gen": {k: {"system": _cap(v.get("system"), 400), "user": _cap(v.get("user"), 800),
                    "raw": _cap(v.get("raw"))}
                for k, v in (cb.gen or {}).items()},
    }


def ingest_sessions(lc: LocomoConversation, n_sessions: int, llm, embedder,
                    atoms: AtomStore, cells: CellStore, ev_store: EvidenceStore):
    """Fast-forward session by session through SessionWriter (W0/W1/W2 in one
    loop, equivalent to the product's online path); returns (time of the last
    session, trace)."""
    trace_sessions = []
    last_dt = None
    for sess in lc.sessions[:n_sessions]:
        writer = SessionWriter(llm, embedder, ev_store, cells, atoms,
                               session_id=f"{lc.sample_id}-s{sess.idx}")
        t0 = time.time()
        ex_records, closed_in_session = [], []
        for ex in sess.exchanges:
            r = writer.feed(ex.holder, ex.text, now_dt=sess.dt)
            ex_records.append({
                "dia": ex.dia, "holder": ex.holder, "text": _cap(ex.text, 800),
                "evidence_id": r.evidence_id,        # dia <-> evidence id, for gold-chain tracing
                "boundary": ({"should_end": r.boundary.should_end,
                              "confidence": r.boundary.confidence,
                              "topic_summary": r.boundary.topic_summary}
                             if r.boundary else None),   # None = first line of a segment (nothing to judge against)
                "forced_close": r.forced_close,      # the 30-turn safety valve, not an LLM decision
            })
            if r.closed_cell:
                closed_in_session.append(r.closed_cell)
        # end_session returns the cell list for the whole session; take only the
        # unrecorded tail, so cells closed mid-session are not counted twice.
        closed_in_session.extend(writer.end_session()[len(closed_in_session):])
        trace_sessions.append({
            "session": sess.idx, "dt": sess.dt.isoformat(),
            "exchanges": ex_records,
            "closed_cells": [_cell_brief(cb) for cb in closed_in_session],
            "cells_total": len(writer.cells),
            "atoms_total": len(atoms.list(limit=100000)),
            "secs": round(time.time() - t0, 1),
        })
        logger.info(f"[{lc.sample_id}] s{sess.idx} done: {len(sess.exchanges)} exchanges -> "
                    f"{len(closed_in_session)} cells / {trace_sessions[-1]['atoms_total']} atoms total "
                    f"({time.time()-t0:.0f}s)")
        last_dt = sess.dt
    return last_dt, trace_sessions


def pick_diverse(qas, n: int) -> list:
    """Sample by rotating through question categories, so every category is covered."""
    by_cat = defaultdict(list)
    for qa in qas:
        by_cat[qa.category].append(qa)
    out = []
    while len(out) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(out) < n:
                out.append(by_cat[cat].pop(0))
    return out


def _atom_view(ah, rank: int) -> dict:
    """Trace view of one hit in the R1 atom pool."""
    return {"rank": rank, "atom_id": ah.atom.id, "text": ah.atom.text,
            "holder": ah.atom.holder, "type": ah.atom.object_type,
            "chain_id": ah.atom.chain_id,
            "rrf": round(ah.rrf, 5), "sim": round(ah.similarity, 4)}


def _hit_view(h, rank: int) -> dict:
    """Trace view of one material unit, shared by the assembly order and the
    reranked order; a non-empty `covers` means a woven memcell."""
    return {
        "rank": rank, "cell_id": h.cell.id, "topic": h.cell.topic,
        "domains": h.cell.domains, "covers": h.covers,
        "rrf": round(h.score, 5), "best_sim": round(h.best_sim, 4),
        "rerank_score": h.rerank_score,
        "atoms": [{"id": a.atom.id, "text": a.atom.text, "holder": a.atom.holder,
                   "type": a.atom.object_type, "sim": round(a.similarity, 4)}
                  for a in h.atoms],
    }


def _qa_trace(o, qa, answer, verdict, r5_verdict, secs, mem_atoms) -> dict:
    """Collect the whole pipeline for one question into the trace, including the
    deep path. The verbatim support for each memory is filled in by main, which
    has the ev_store.

    Two conventions: judge is the Mem0 convention (the answerer's output), and
    judge_r5 is the product convention (the R5 answer judged directly)."""
    memories = [{"atom_id": a.id, "text": stamped_atom_text(a), "type": a.object_type,
                 "evidence": []} for a in mem_atoms]
    return {
        "category": qa.category, "question": qa.question, "gold": qa.answer,
        "evidence": qa.evidence, "secs": secs,
        "rewrite": {"resolved": o.rw.resolved if o.rw else None,
                    "subject": o.rw.subject if o.rw else "",
                    "expansions": o.rw.expansions if o.rw else [],
                    "time_window": ([o.rw.time_start, o.rw.time_end]
                                    if (o.rw and (o.rw.time_start or o.rw.time_end)) else []),
                    "domains": o.rw.domains if o.rw else []},
        "pool": [_atom_view(a, i + 1) for i, a in enumerate(o.hits[:10])],     # the R1 atom pool
        "hits": [_hit_view(h, i + 1) for i, h in
                 enumerate((o.asm.units if o.asm else [])[:10])],               # material units, in assembly order
        "ranked": [_hit_view(h, i + 1) for i, h in enumerate(o.ranked[:10])],  # the R2 reranked order
        "review": ({"verdict": o.reviews[-1].verdict, "critique": o.reviews[-1].critique,
                    "retried": o.retried} if o.reviews else None),
        "r5": {"answer": o.ans.answer, "cited_cells": o.ans.cited_cells} if o.ans else None,
        "deep": ({"steps": o.deep.steps, "remembered": o.deep.remembered,
                  "escalated": o.escalated, "answer": o.deep.ans.answer,
                  "cited_cells": o.deep.ans.cited_cells}
                 if o.deep else None),                                       # the deep-path trace; None if it did not run
        "memories": memories,
        "answer": answer, "judge": verdict.ok, "judge_raw": verdict.raw,
        "judge_r5": r5_verdict.ok, "judge_r5_raw": r5_verdict.raw,
    }


def _dia_evidence(lc, store_evidence: list) -> dict:
    """dia -> evidence id: align the adapter's dia values with the evidence in
    the store by utterance text prefix.

    A fallback used only under --skip-ingest, when there is no ingest record.
    For a merged turn, the prefix of any line after the first does not match the
    prefix of the whole evidence record, so some mappings may be missed. When an
    ingest record exists, main uses the exact dia -> evidence_id mapping instead.
    """
    prefix2ev = {}
    for e in store_evidence:
        c = e.get("content") or ""
        if c:
            prefix2ev.setdefault(c[:60], e["id"])
    out = {}
    for s in lc.sessions:
        for ex in s.exchanges:
            dias = [d for d in (ex.dia or "").split(",") if d]
            for d, line in zip(dias, ex.text.split("\n")):
                ev_id = prefix2ev.get(line[:60])
                if ev_id:
                    out[d] = ev_id
    return out


def _gold_chain(qa_rec: dict, dia2ev: dict, store_atoms: list) -> list:
    """The gold evidence chain: how far each verbatim line (dia) the gold answer
    points at travelled through this question's pipeline.

    The stages: evidence (did it get stored?) -> extraction (was an atom
    extracted from it?) -> retrieval (did it reach the R1 top 10 cells?) ->
    materials (did it reach the top 20 R5 answering materials?) -> adoption (did
    R5 cite that cell?).
    The break point is the first stage that failed, i.e. where the error is.
    """
    hit_atom_ids = {a["id"] for h in qa_rec.get("hits", []) for a in h["atoms"]}
    ranked_atom_ids = {a["id"] for h in qa_rec.get("ranked", []) for a in h["atoms"]}
    cited = set((qa_rec.get("r5") or {}).get("cited_cells") or [])
    ev2atoms: dict[str, list] = defaultdict(list)
    for a in store_atoms:
        for e in a.get("evidence") or []:
            ev2atoms[e].append(a)
    chains = []
    for d in qa_rec.get("evidence", []):
        ev_id = dia2ev.get(d)
        hit_atoms = ev2atoms.get(ev_id, []) if ev_id else []
        chains.append({
            "dia": d, "ev_id": ev_id,
            "n_extracted": len(hit_atoms),
            "extracted_atoms": [a["text"][:60] for a in hit_atoms[:3]],
            "cell_ids": sorted({a["cell_id"] for a in hit_atoms if a.get("cell_id")}),
            "in_hits": any(a["id"] in hit_atom_ids for a in hit_atoms),
            "in_material": any(a["id"] in ranked_atom_ids for a in hit_atoms),
            "cited": bool(hit_atoms and (cited & {a["cell_id"] for a in hit_atoms})),
        })
    return chains


def _break_point(qa_rec: dict, chains: list) -> str:
    """Determine the break point: how far the pipeline could get, with the error
    lying just beyond that stage."""
    if not chains:
        return "no evidence (the gold answer points nowhere)"
    if not any(c["ev_id"] for c in chains):
        return "evidence (the dia was never stored)"
    if not any(c["n_extracted"] for c in chains):
        return "extraction (the verbatim text is stored but no atom was extracted)"
    if any(c["cited"] for c in chains):
        # The gold evidence was cited by R5 and it is still wrong, so it died in
        # answering or scoring.
        ans = (qa_rec.get("answer") or "")
        if ans.lower().startswith("i don't have"):
            return "answerer (refused despite citing the evidence)"
        return "answering (the answerer drifted or the judge misjudged; review by hand)"
    if any(c["in_material"] for c in chains):
        return "adoption (reached the answering materials but R5 did not cite it)"
    if any(c["in_hits"] for c in chains):
        return "materials (retrieved but cut off by the top-20 limit)"
    return "retrieval (the atom is stored but neither path recalled it)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--n-sessions", type=int, default=3)
    ap.add_argument("--n-questions", type=int, default=8)
    ap.add_argument("--out", default="data/bench/locomo_demo")
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "deep"])
    ap.add_argument("--data", default=str(LOCOMO_PATH),
                    help="path to locomo10.json (default data/locomo10.json; override it when "
                         "running from a persistent volume or a GPU server)")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse the user data already loaded for this conversation and run only "
                         "the question stage, saving load time while debugging scoring or retrieval")
    ap.add_argument("--ingest-only", action="store_true",
                    help="load the store and stop without answering; for batch runs, load several "
                         "conversations concurrently first and then answer them all with --skip-ingest")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="how many questions of one conversation are answered in parallel "
                         "(default 1 = serial; each question gets its own session_id, so they "
                         "cannot contaminate each other)")
    ap.add_argument("--questions-file", default="",
                    help="a JSON array file of exact question texts; run only those questions, "
                         "bypassing pick_diverse sampling. Useful for small regressions.")
    args = ap.parse_args()

    convs = load_conversations(args.data)
    lc = next(c for c in convs if c.sample_id == args.conv)
    print(f"conversation {lc.sample_id}: {len(lc.sessions)} sessions / {len(lc.qa)} questions; "
          f"taking the first {args.n_sessions} sessions")

    # Each conversation gets its own user namespace, and a re-run clears that
    # user's four tables first (idempotent, so atom counts do not double).
    uid = f"locomo-{lc.sample_id}"
    db = Database()
    have = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
    if args.skip_ingest:
        assert have > 0, f"--skip-ingest needs user data to already be loaded: {uid}"
    elif have:
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
    ev_store, atoms = EvidenceStore(db, uid), AtomStore(db, uid)
    cells = CellStore(db, uid)
    llm = AnthropicChatLLM()      # the benchmark LLM: W1/W2/R0/R3/R5/answerer/judge
    embedder = OpenAIEmbedder()   # embeddings: topic/atom vectors on write, and the query side on read
    scorer = OpenAIReranker()     # the reranking endpoint driving the R2 stage

    trace: dict = {"meta": {"conv": lc.sample_id, "n_sessions": args.n_sessions,
                            "mode": args.mode, "llm": "MiniMax-M3",
                            "judge": "MiniMax-M3 (Mem0-style binary, v2 with relative-time conversion)",
                            "answerer": "mode A, two conventions (brief = the R5 answer; memories = "
                                        "atoms hit in the top 20 reranked cells, grouped by cell, "
                                        "under trust-the-brief; judge_r5 scores R5 directly)",
                            "chain": "step 1 write + step 2 fast path",
                            "run_at": time.strftime("%Y-%m-%d %H:%M"), "data": args.data}}
    try:   # the short git hash goes into meta so a report traces back to a code version
        import subprocess
        trace["meta"]["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    # -- Load the store (SessionWriter: W0 evidence / W1 boundaries / W2 cells) --
    if args.skip_ingest:
        # n_sessions is usually passed as 99 to mean "all of them"; the question
        # time is the last session actually loaded.
        last_dt = lc.sessions[min(args.n_sessions, len(lc.sessions)) - 1].dt
        trace["ingest"] = []
        print(f"reusing already-loaded user {uid} ({len(atoms.list(limit=100000))} atoms / "
              f"{len(cells.iter_all())} cells), asking as of {last_dt.isoformat()}")
    else:
        t0 = time.time()
        last_dt, trace["ingest"] = ingest_sessions(lc, args.n_sessions, llm, embedder,
                                                   atoms, cells, ev_store)
        print(f"\nload complete ({time.time()-t0:.0f}s):")
        for s in trace["ingest"]:
            cc = s["closed_cells"]
            print(f"  s{s['session']} @{s['dt'][:10]}: {len(s['exchanges'])} exchanges -> "
                  f"{len(cc)} cells / {s['atoms_total']} atoms total ({s['secs']}s)")

    # Load only: write a load trace for inspection and stop before answering, so
    # a batch can fill several conversations first and answer them together.
    if args.ingest_only:
        n_atoms = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
        n_cells = db.fetch_one("SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s", (uid,))["n"]
        print(f"\n[ingest-only] {lc.sample_id} loaded: user={uid} atoms={n_atoms} cells={n_cells}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix(".ingest.json")).write_text(
            json.dumps({"meta": trace["meta"], "ingest": trace["ingest"],
                        "atoms": n_atoms, "cells": n_cells}, ensure_ascii=False, indent=1))
        print(f"load trace written to {out.with_suffix('.ingest.json')}")
        return

    # -- Select questions whose evidence lies entirely in the loaded sessions --
    sidx = set(range(1, args.n_sessions + 1))
    answerable = pick_answerable(lc, sidx)
    if args.questions_file:
        wanted = [q.strip() for q in json.loads(
            Path(args.questions_file).read_text(encoding="utf-8"))]
        by_text = {q.question.strip(): q for q in answerable}
        qas = [by_text[w] for w in wanted if w in by_text]
        miss = len(wanted) - len(qas)
        if miss:
            print(f"warning: {miss}/{len(wanted)} questions did not match (outside the evidence "
                  f"range, or the question text differs)")
    else:
        qas = pick_diverse(answerable, args.n_questions)
    print(f"\n{len(answerable)} answerable questions (all evidence within the first "
          f"{args.n_sessions} sessions, cat5 excluded); answering {len(qas)} of them\n")

    # -- Per question: the whole fast path -> answerer -> judge, N at a time --
    reranker = ScoringReranker(scorer)   # the real R2 reranking stage, same as the product /recall

    def answer_one(i: int, qa) -> dict:
        """Answer and score one question, returning its trace record.

        Concurrency-safe because each question gets its own session_id
        ({conv}-qa-{i}), so they cannot contaminate each other's build_history
        or session_context; atoms/cells/evidence are read-only (the DB engine's
        connection pool is thread-safe); and the LLM clients hold no shared
        mutable state.
        """
        t1 = time.time()
        for attempt in range(1, 4):   # retry 3x per question, 60s apart, so one network blip does not sink the round
            try:
                o = run_recall(llm, embedder, atoms, cells, ev_store,
                               session_id=f"{lc.sample_id}-qa-{i}", query=qa.question,
                               now_dt=last_dt, mode=args.mode, reranker=reranker)
                if o.deep and not o.ranked:
                    groups = [(c, atoms.list_by_cell(c.id))
                              for c in (cells.get(cid) for cid in (o.ans.cited_cells
                                                                   if o.ans else [])) if c]
                else:
                    groups = [(h.cell, [sa.atom for sa in h.atoms])
                              for h in o.ranked[:_ANSWER_MEM_CELLS]]
                mem_atoms = [a for _, atom_list in groups for a in atom_list]
                answer = answer_mode_a(llm, question=qa.question, brief=o.ans.answer,
                                       mem_block=_grouped_materials(groups))
                v = judge(llm, question=qa.question, gold=qa.answer, prediction=answer)
                try:
                    v5 = judge(llm, question=qa.question, gold=qa.answer,
                               prediction=o.ans.answer if o.ans else "")
                except Exception:   # noqa: BLE001
                    logger.exception("the product-convention judge failed")
                    v5 = Verdict(ok=False, raw="ERROR: judge_r5 failed (see the run log)")
                rec = _qa_trace(o, qa, answer, v, v5, round(time.time() - t1, 1), mem_atoms)
                for m, a in zip(rec["memories"], mem_atoms):
                    m["evidence"] = [{"holder": e["holder"], "content": _cap(e["content"], 400),
                                      "at": (e.get("captured_at") or "")[:10]}
                                     for e in evidence_entries(a, ev_store)]
                verdict = o.reviews[-1].verdict if o.reviews else "?"
                deep_part = f", deep={len(o.deep.steps)} steps" if o.deep else ""
                print(f"[{i}/{len(qas)}] {'ok' if v.ok else 'X'}/{'ok' if v5.ok else 'X'} "
                      f"cat{qa.category} ({rec['secs']}s, review={verdict}{deep_part}) "
                      f"Q: {qa.question[:60]}", flush=True)
                return rec
            except Exception:   # noqa: BLE001
                logger.exception(f"Q{i} failed, attempt {attempt}/3")
                if attempt < 3:
                    time.sleep(60)
        print(f"[{i}/{len(qas)}] X/X cat{qa.category} (failed after 3 retries) "
              f"Q: {qa.question[:60]}", flush=True)
        return {"category": qa.category, "question": qa.question, "gold": qa.answer,
                "evidence": qa.evidence, "secs": round(time.time() - t1, 1),
                "memories": [], "answer": "", "judge": False,
                "judge_raw": "ERROR: still failing after 3 retries (see the run log)",
                "judge_r5": False, "judge_r5_raw": "ERROR: still failing after 3 retries"}

    # Thread pool; results are filled back in by question number so that the
    # order of qa_records matches a serial run and stays reproducible/diffable.
    results: dict[int, dict] = {}
    conc = max(1, args.concurrency)
    print(f"answering concurrency: {conc}\n", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(answer_one, i, qa): i for i, qa in enumerate(qas, 1)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    qa_records = [results[i] for i in sorted(results)]

    # -- The whole store, in its final state after the benchmark --
    trace["store"] = {
        "cells": [{"id": c.id, "session": c.session_id, "topic": c.topic,
                   "domains": c.domains,
                   "t_start": c.t_start.isoformat()[:19] if c.t_start else "",
                   "t_end": c.t_end.isoformat()[:19] if c.t_end else "",
                   "episode": _cap(c.episode, 400),
                   "n_atoms": len(atoms.list_by_cell(c.id))}
                  for c in cells.iter_all(limit=10000)],
        "atoms": [_atom_brief(a) for a in atoms.list(limit=100000)],
        "evidence": [{"id": e.id, "holder": e.holder,
                      "at": e.captured_at.isoformat()[:19] if e.captured_at else "",
                      "content": _cap(e.content_inline, 500)}
                     for e in ev_store.iter_all()],
    }
    # -- Gold evidence chain and break point, so that every question (above all
    #    every wrong one) can be pinned to the stage where it went wrong --
    if trace["ingest"]:
        # Just loaded: the ingest record maps dia to evidence_id exactly (the
        # several dia values of a merged turn share one evidence record).
        dia2ev = {d: ex["evidence_id"]
                  for s in trace["ingest"] for ex in s["exchanges"]
                  for d in filter(None, (ex["dia"] or "").split(","))}
    else:   # --skip-ingest: alignment by text prefix is all that is available
        dia2ev = _dia_evidence(lc, trace["store"]["evidence"])
    for q in qa_records:
        q["gold_chain"] = _gold_chain(q, dia2ev, trace["store"]["atoms"])
        q["break_point"] = _break_point(q, q["gold_chain"])
    n_ok = sum(r["judge"] for r in qa_records)
    n_ok5 = sum(bool(r.get("judge_r5")) for r in qa_records)   # the product convention
    by_cat = defaultdict(lambda: [0, 0])
    for r in qa_records:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["judge"])
    verdicts = defaultdict(int)
    for r in qa_records:
        verdicts[(r.get("review") or {}).get("verdict", "?")] += 1
    trace["qa"] = qa_records
    trace["summary"] = {"score": n_ok, "score_r5": n_ok5, "n_questions": len(qa_records),
                        "by_category": {str(k): v for k, v in sorted(by_cat.items())},
                        "verdicts": dict(verdicts),
                        "retried": sum(bool((r.get("review") or {}).get("retried")) for r in qa_records),
                        "escalated": sum(bool((r.get("deep") or {}).get("escalated"))
                                         for r in qa_records)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".trace.json")).write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    n = max(1, len(qa_records))
    print(f"\n===== result: Mem0 convention (answerer) {n_ok}/{len(qa_records)} "
          f"({100*n_ok/n:.0f}%) - product convention (R5 judged directly) "
          f"{n_ok5}/{len(qa_records)} ({100*n_ok5/n:.0f}%) =====")
    for cat, (ok, tot) in sorted(by_cat.items()):
        print(f"  cat{cat}: {ok}/{tot}")
    print(f"review verdicts: {dict(verdicts)}")
    print(f"trace written to {out.with_suffix('.trace.json')}")

    # Render the full-pipeline HTML report and the final markdown report; both
    # are self-contained and open directly in a browser or editor.
    from scripts.bench.report import render, render_md
    html_path = out.with_suffix(".report.html")
    render(trace, html_path)
    md_path = out.with_suffix(".report.md")
    render_md(trace, md_path)
    print(f"full-pipeline report: {html_path}\n"
          f"final report (total, per category, error analysis): {md_path}")
    embedder.close(); scorer.close(); llm.close(); db.close()


if __name__ == "__main__":
    main()
