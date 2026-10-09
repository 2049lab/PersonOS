"""LongMemEval-S x personos benchmark, self-contained: download -> per-question
ingest -> answer -> official judge -> per-question trace on disk.

The benchmark is structurally different from LoCoMo-10: every question is its
own world. LongMemEval-S stitches a synthetic user history (~40 sessions,
~500 turns, sampled from ShareGPT / UltraChat) and asks one question on top;
500 questions in total. Each question owns its own user_id (`lme-<qid>`),
its own haystack, and its own answer trace, so progress and re-runs are
fully decoupled. The dataset deliberately reuses real session_ids across
questions, which makes per-question user namespaces both correct (the
"synthetic user" for question A is not the same person as for question B) and
the only way to ingest without races.

Official protocol (https://github.com/xiaowu0162/LongMemEval,
src/evaluation/evaluate_qa.py) is followed verbatim:

- judge prompts are copied character-for-character from `get_anscheck_prompt`,
  five question_type branches plus the abstention branch (`_abs` suffix on
  question_id). The judge is a yes/no binary; `label = 'yes' in eval_response.lower()`.
- abstention questions (30 in LME-S) flip the prompt's "Correct Answer" to
  an "Explanation" and ask the model to identify the question as unanswerable.
- scoring is reported per `question_type` and as an overall accuracy.

Resumability / multi-day runs:

- each question is independent: a question that already has a trace JSON in
  the run dir is skipped entirely; a question whose user namespace already
  has atoms is re-ingested only on explicit `--force-reingest`, otherwise the
  existing memory is reused and only the answer+judge are re-run.
- `--questions-file` lets a long run be split into shards; pass the same
  `--run-dir` across days so traces accumulate.
- a live `progress.jsonl` is appended to after every question completes
  (one line: question_id, task, abstention, score, secs, status), so a cron
  watcher can see where the run is without parsing traces.

Artifacts (git-ignored, under data/bench/runs/<run_id>/):
- <qid>.trace.json   full pipeline trace per question (rewrite/hits/ranked/review/deep)
- <qid>.report.md    short per-question report (totals + per-task)
- progress.jsonl     live progress log (one row per completed question)
- summary.json / summary.md   run-level aggregate

Usage:
  python -m scripts.bench.longmemeval --download
  python -m scripts.bench.longmemeval --questions-file data/bench/shard-a.jsonl
  python -m scripts.bench.longmemeval --all --concurrency 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from personos.config import get_config
from personos.models import TZ, stamped_atom_text
from personos.online.recall_flow import run_recall
from personos.online.rerank import NoopReranker, ScoringReranker
from personos.online.retrieval import cell_lead
from personos.online.trust import evidence_entries
from personos.online.write_path import SessionWriter
from personos.providers.registry import build as build_provider
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore

LME_PATH = Path("data/longmemeval_s_cleaned.json")    # --download fills it; --data overrides
# The official home is the xiaowu0162/longmemeval-cleaned HF dataset. The
# cleaned _s variant is ~500 questions, ~40 haystack sessions each, ~115k
# tokens of context per question. huggingface_hub is imported lazily and kept
# out of the package dependencies on purpose (a community install per LoCoMo).
HF_REPO = "xiaowu0162/longmemeval-cleaned"
HF_FILE = "longmemeval_s_cleaned.json"

# Truncation limit for long text in the trace (the complete verbatim text lives
# in the store and is never cut).
_CAP_RAW = 2000

# How wide the answerer's memory view is: same as LoCoMo's _ANSWER_MEM_CELLS,
# 20 cells (the size R5 itself sees, so answerer and R5 stay aligned).
_ANSWER_MEM_CELLS = 20


# ═══════════════════════════ data adapter ═══════════════════════════════
# Pure parsing; no LLM or storage calls. LME timestamps look like
# "2023/05/20 (Sat) 02:21" — the weekday is informational, not authoritative;
# Python's strptime doesn't accept it natively.

_DT_RE = re.compile(
    r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})\s+\([A-Za-z]+\)\s+(?P<H>\d{2}):(?P<M>\d{2})$"
)


@dataclass
class HaystackSession:
    sid: str
    dt: datetime
    turns: list[dict]   # [{"role": "user"/"assistant", "content": str}, ...]


@dataclass
class LmeInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_dt: datetime
    sessions: list[HaystackSession]   # the haystack in dataset order
    answer_session_ids: list[str] = field(default_factory=list)
    is_abstention: bool = False       # question_id contains "_abs"

    def answer_reachable(self, n_sessions: int) -> bool:
        """Whether at least one answer_session_id falls inside the first
        n_sessions of the haystack, so a question that is gated by --n-sessions
        is only asked when its evidence has actually been ingested."""
        sids = {s.sid for s in self.sessions[:n_sessions]}
        return any(s in sids for s in self.answer_session_ids)


def _parse_dt(raw: str) -> datetime:
    m = _DT_RE.match((raw or "").strip())
    if not m:
        raise ValueError(f"cannot parse LME timestamp: {raw!r}")
    return datetime(int(m["y"]), int(m["m"]), int(m["d"]),
                    int(m["H"]), int(m["M"]), tzinfo=TZ)


def load_dataset(path: str | Path) -> list[LmeInstance]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) > 0 and all(
        "question_id" in x and "haystack_sessions" in x for x in raw
    ), f"{path} does not look like longmemeval_s_cleaned.json"
    out: list[LmeInstance] = []
    for x in raw:
        sids = x["haystack_session_ids"]
        dates = x["haystack_dates"]
        sessions = [
            HaystackSession(sid=sid, dt=_parse_dt(dt_raw), turns=turns)
            for sid, dt_raw, turns in zip(sids, dates, x["haystack_sessions"])
        ]
        out.append(LmeInstance(
            question_id=x["question_id"],
            question_type=x["question_type"],
            question=x["question"],
            answer=x["answer"],
            question_dt=_parse_dt(x["question_date"]),
            sessions=sessions,
            answer_session_ids=x.get("answer_session_ids", []),
            is_abstention="_abs" in x["question_id"],
        ))
    return out


def download_dataset(dest: Path = LME_PATH) -> Path:
    """Fetch longmemeval_s_cleaned.json from the official Hugging Face dataset.
    huggingface_hub is imported lazily and kept out of package dependencies."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
        shutil.copy(src, dest)
        origin = f"huggingface:{HF_REPO}"
    except Exception as e:
        raise SystemExit(
            f"failed to download {HF_FILE} from {HF_REPO}: "
            f"{type(e).__name__}: {e}\n"
            f"if the dataset is gated in your region, try a local copy and pass --data <path>."
        )
    instances = json.loads(dest.read_text(encoding="utf-8"))
    assert isinstance(instances, list) and len(instances) >= 400 and all(
        "question_id" in x and "haystack_sessions" in x for x in instances
    ), f"{dest} does not look like longmemeval_s_cleaned.json"
    abst = sum(1 for x in instances if "_abs" in x["question_id"])
    print(f"dataset downloaded from {origin}: {dest} ({dest.stat().st_size / 1e6:.1f} MB, "
          f"{len(instances)} questions, {abst} abstention)")
    return dest


# ═══════════════════════════ judge (official, verbatim) ═════════════════
# The judge prompt descends from xiaowu0162/LongMemEval
# src/evaluation/evaluate_qa.py:get_anscheck_prompt. Five question_type
# branches plus an abstention branch. These prompts ARE the protocol — copy
# verbatim, never edit, never paraphrase. The judge model is written into the
# run meta so a report traces back to its scorer.

@dataclass
class Verdict:
    ok: bool
    raw: str


# Three of the six question_types ('single-session-user',
# 'single-session-assistant', 'multi-session') share one template in the
# official code; the others each have their own. Each branch maps verbatim.
_ANSCHECK_TEMPLATES = {
    "shared": (
        "I will give you a question, a correct answer, and a response from a model. Please answer "
        "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
        "equivalent to the correct answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a subset of the "
        "information required by the answer, answer no. \n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. Please answer "
        "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
        "equivalent to the correct answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a subset of the "
        "information required by the answer, answer no. In addition, do not penalize off-by-one "
        "errors for the number of days. If the question asks for the number of days/weeks/months, "
        "etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is "
        "18), the model's response is still correct. \n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer "
        "yes if the response contains the correct answer. Otherwise, answer no. If the response "
        "contains some previous information along with an updated answer, the response should be "
        "considered as correct as long as the updated answer is the required answer.\n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
        "answer no. The model does not need to reflect all the points in the rubric. The response "
        "is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
        "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The model "
        "could say that the information is incomplete, or some other information is given but the "
        "asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def judge_lme(judge_llm, *, task: str, question: str, gold: str, prediction: str,
              abstention: bool = False) -> Verdict:
    """Official LME yes/no judge (six question_types + abstention). The prompt
    is verbatim from evaluate_qa.py:get_anscheck_prompt — the prompts are the
    protocol, never edit."""
    if abstention:
        prompt = _ANSCHECK_TEMPLATES["abstention"].format(question, gold, prediction)
    elif task == "temporal-reasoning":
        prompt = _ANSCHECK_TEMPLATES["temporal-reasoning"].format(question, gold, prediction)
    elif task == "knowledge-update":
        prompt = _ANSCHECK_TEMPLATES["knowledge-update"].format(question, gold, prediction)
    elif task == "single-session-preference":
        prompt = _ANSCHECK_TEMPLATES["single-session-preference"].format(question, gold, prediction)
    elif task in ("single-session-user", "single-session-assistant", "multi-session"):
        prompt = _ANSCHECK_TEMPLATES["shared"].format(question, gold, prediction)
    else:
        raise NotImplementedError(f"unknown LME task: {task!r}")
    raw = judge_llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=16,
    ).strip()
    # Official parser: label = 'yes' in eval_response.lower()
    ok = "yes" in raw.lower()
    return Verdict(ok=ok, raw=raw)


# ═══════════════════════════ per-instance ingest ═════════════════════════
# Each instance is one synthetic user; every haystack_session is fed in
# sequence (one SessionWriter.feed_batch per session so the session's
# segmentation is preserved), then the answerer asks with now_dt set to the
# question's question_date — the exact protocol the LME benchmark expects of
# a memory-augmented system.

def _speaker_of(role: str) -> str:
    # LME sessions are user/assistant turns; the human is "user", the model
    # is "assistant". We pass "user" straight through as the holder; assistant
    # turns are kept in evidence as the model's replies to the human, which
    # the extraction pipeline can either pick up or discard on its own.
    return "user" if role == "user" else "assistant"


def ingest_instance(inst: LmeInstance, *, n_sessions: int, llm, embedder,
                    atoms: AtomStore, cells: CellStore, ev_store: EvidenceStore,
                    trace_out: list) -> datetime:
    """Feed every haystack session of one instance into the store (W0 evidence
    / W1 boundaries / W2 cell building, with now_dt set to the session time).
    Returns the timestamp of the last loaded session (the question's natural
    'now' baseline)."""
    last_dt = inst.sessions[0].dt
    for sess in inst.sessions[:n_sessions]:
        writer = SessionWriter(llm, embedder, ev_store, cells, atoms,
                               session_id=f"lme-{inst.question_id}-{sess.sid}")
        t0 = time.time()
        ex_records, closed_in_session = [], []
        for turn in sess.turns:
            speaker = _speaker_of(turn["role"])
            r = writer.feed(speaker, turn["content"], now_dt=sess.dt)
            ex_records.append({
                "role": turn["role"], "text": _cap(turn["content"], 600),
                "evidence_id": r.evidence_id,
                "boundary": ({"should_end": r.boundary.should_end,
                              "confidence": r.boundary.confidence,
                              "topic_summary": r.boundary.topic_summary}
                             if r.boundary else None),
            })
            if r.closed_cell:
                closed_in_session.append(r.closed_cell)
        closed_in_session.extend(writer.end_session()[len(closed_in_session):])
        trace_out.append({
            "sid": sess.sid, "dt": sess.dt.isoformat(),
            "n_turns": len(sess.turns),
            "exchanges": ex_records,
            "closed_cells": len(closed_in_session),
            "secs": round(time.time() - t0, 1),
        })
        logger.info(f"[{inst.question_id}] s={sess.sid} done: {len(sess.turns)} turns -> "
                    f"{len(closed_in_session)} cells ({time.time()-t0:.0f}s)")
        last_dt = sess.dt
    return last_dt


# ═══════════════════════════ trace views ═════════════════════════════════

def _cap(s: str | None, n: int = _CAP_RAW) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f" ...[truncated; full text is {len(s)} chars]"


def _atom_view(ah, rank: int) -> dict:
    return {"rank": rank, "atom_id": ah.atom.id, "text": ah.atom.text,
            "holder": ah.atom.holder, "type": ah.atom.object_type,
            "chain_id": ah.atom.chain_id,
            "rrf": round(ah.rrf, 5), "sim": round(ah.similarity, 4)}


def _hit_view(h, rank: int) -> dict:
    return {
        "rank": rank, "cell_id": h.cell.id, "topic": h.cell.topic,
        "domains": h.cell.domains, "covers": h.covers,
        "rrf": round(h.score, 5), "best_sim": round(h.best_sim, 4),
        "rerank_score": h.rerank_score,
        "atoms": [{"id": a.atom.id, "text": a.atom.text, "holder": a.atom.holder,
                   "type": a.atom.object_type, "sim": round(a.similarity, 4)}
                  for a in h.atoms],
    }


def _qa_trace(o, inst: LmeInstance, secs: float, mem_atoms: list) -> dict:
    memories = [{"atom_id": a.id, "text": stamped_atom_text(a), "type": a.object_type,
                 "evidence": []} for a in mem_atoms]
    return {
        "question_id": inst.question_id, "task": inst.question_type,
        "abstention": inst.is_abstention,
        "question": inst.question, "gold": inst.answer, "secs": secs,
        "rewrite": {"resolved": o.rw.resolved if o.rw else None,
                    "subject": o.rw.subject if o.rw else "",
                    "expansions": o.rw.expansions if o.rw else [],
                    "time_window": ([o.rw.time_start, o.rw.time_end]
                                    if (o.rw and (o.rw.time_start or o.rw.time_end)) else []),
                    "domains": o.rw.domains if o.rw else []},
        "pool": [_atom_view(a, i + 1) for i, a in enumerate(o.hits[:10])],
        "hits": [_hit_view(h, i + 1) for i, h in
                 enumerate((o.asm.units if o.asm else [])[:10])],
        "ranked": [_hit_view(h, i + 1) for i, h in enumerate(o.ranked[:10])],
        "review": ({"verdict": o.reviews[-1].verdict, "critique": o.reviews[-1].critique,
                    "retried": o.retried} if o.reviews else None),
        "r5": {"answer": o.ans.answer, "cited_cells": o.ans.cited_cells} if o.ans else None,
        "deep": ({"steps": o.deep.steps, "remembered": o.deep.remembered,
                  "escalated": o.escalated, "answer": o.deep.ans.answer,
                  "cited_cells": o.deep.ans.cited_cells}
                 if o.deep else None),
        "memories": memories,
    }


def _grouped_materials(groups: list[tuple]) -> str:
    """Group by cell, each group headed by cell_lead (dialogue time plus topic)
    and followed by stamped atoms. Structurally identical to LoCoMo's
    _grouped_materials so the answerer sees the same shape of context."""
    out = []
    for c, atom_list in groups:
        lines = "\n".join(f"- {stamped_atom_text(a)}" for a in atom_list)
        out.append(f"{cell_lead(c)}\n{lines}" if lines else cell_lead(c))
    return "\n\n".join(out)


# ═══════════════════════════ per-instance run ═══════════════════════════

def run_instance(inst: LmeInstance, args, run_dir: Path, llm, judge_llm,
                 embedder, reranker, db: Database, meta: dict) -> dict | None:
    """One question, end to end. Resumable: a question that already has a
    trace file is skipped; a question whose user already has atoms is
    re-ingested only on explicit --force-reingest.

    Returns the per-question trace dict, or None if the question was skipped
    because its trace was already on disk.
    """
    trace_path = run_dir / f"{inst.question_id}.trace.json"
    if trace_path.exists() and not args.force_reanswer:
        print(f"[{inst.question_id}] SKIP: trace already exists ({trace_path.name})")
        return None

    uid = f"lme-{inst.question_id}"
    have_atoms = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
    if args.force_reingest and have_atoms:
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
        have_atoms = 0
        print(f"[{inst.question_id}] force-reingest: cleared existing store for {uid}")
    ev_store, atoms = EvidenceStore(db, uid), AtomStore(db, uid)
    cells = CellStore(db, uid)
    print(f"\n===== {inst.question_id} [{inst.question_type}]"
          f"{' (abstention)' if inst.is_abstention else ''}: "
          f"{len(inst.sessions)} sessions / {sum(len(s.turns) for s in inst.sessions)} turns =====")

    trace: dict = {"meta": {**meta, "question_id": inst.question_id,
                            "task": inst.question_type,
                            "abstention": inst.is_abstention,
                            "n_sessions": min(args.n_sessions, len(inst.sessions))}}

    # -- Ingest the haystack --
    if have_atoms:
        last_dt = inst.sessions[min(args.n_sessions, len(inst.sessions)) - 1].dt
        trace["ingest"] = []
        trace["ingest_skipped"] = (have_atoms, len(cells.iter_all()))
        print(f"reusing loaded user {uid} ({have_atoms} atoms / "
              f"{len(cells.iter_all())} cells), asking as of {last_dt.isoformat()}")
    else:
        t0 = time.time()
        sessions_to_load = inst.sessions[:args.n_sessions]
        ingest_trace: list = []
        last_dt = ingest_instance(inst, n_sessions=args.n_sessions, llm=llm,
                                  embedder=embedder, atoms=atoms, cells=cells,
                                  ev_store=ev_store, trace_out=ingest_trace)
        trace["ingest"] = ingest_trace
        n_atoms = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
        n_cells = db.fetch_one("SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s", (uid,))["n"]
        print(f"load complete ({time.time()-t0:.0f}s, "
              f"{len(sessions_to_load)} sessions / "
              f"{sum(len(s.turns) for s in sessions_to_load)} turns -> "
              f"{n_atoms} atoms / {n_cells} cells)")

    # -- Answer + judge --
    t1 = time.time()
    for attempt in range(1, 4):
        try:
            o = run_recall(llm, embedder, atoms, cells, ev_store,
                           session_id=f"lme-{inst.question_id}-qa",
                           query=inst.question, now_dt=inst.question_dt,
                           mode=args.mode, reranker=reranker)
            if o.deep and not o.ranked:
                groups = [(c, atoms.list_by_cell(c.id))
                          for c in (cells.get(cid) for cid in (o.ans.cited_cells
                                                               if o.ans else [])) if c]
            else:
                groups = [(h.cell, [sa.atom for sa in h.atoms]) for h in o.ranked[:_ANSWER_MEM_CELLS]]
            mem_atoms = [a for _, atom_list in groups for a in atom_list]
            secs = round(time.time() - t1, 1)
            rec = _qa_trace(o, inst, secs, mem_atoms)
            for m, a in zip(rec["memories"], mem_atoms):
                m["evidence"] = [{"holder": e["holder"], "content": _cap(e["content"], 400),
                                  "at": (e.get("captured_at") or "")[:10]}
                                 for e in evidence_entries(a, ev_store)]
            prediction = (o.ans.answer if o.ans else "") or ""
            v = judge_lme(judge_llm, task=inst.question_type, question=inst.question,
                          gold=inst.answer, prediction=prediction, abstention=inst.is_abstention)
            rec["answer"] = prediction
            rec["judge"] = v.ok
            rec["judge_raw"] = v.raw
            verdict = o.reviews[-1].verdict if o.reviews else "?"
            deep_part = f", deep={len(o.deep.steps)} steps" if o.deep else ""
            tag = "ABST" if inst.is_abstention else ("ok" if v.ok else "X")
            print(f"[{inst.question_id}] {tag}/{v.raw[:20]} task={inst.question_type} "
                  f"({secs}s, review={verdict}{deep_part}) Q: {inst.question[:60]}", flush=True)
            trace["qa"] = [rec]
            trace["summary"] = {"score": int(v.ok), "task": inst.question_type,
                                "abstention": inst.is_abstention,
                                "secs": secs}
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1))
            return trace
        except Exception:   # noqa: BLE001
            logger.exception(f"[{inst.question_id}] Q failed, attempt {attempt}/3")
            if attempt < 3:
                time.sleep(60)
    trace["qa"] = [{"question_id": inst.question_id, "task": inst.question_type,
                    "abstention": inst.is_abstention,
                    "question": inst.question, "gold": inst.answer,
                    "secs": round(time.time() - t1, 1),
                    "answer": "", "judge": False,
                    "judge_raw": "ERROR: still failing after 3 retries (see the run log)"}]
    trace["summary"] = {"score": 0, "task": inst.question_type,
                        "abstention": inst.is_abstention,
                        "secs": round(time.time() - t1, 1)}
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    print(f"[{inst.question_id}] X/X task={inst.question_type} (failed after 3 retries)",
          flush=True)
    return trace


# ═══════════════════════════ entry ══════════════════════════════════════

def _progress_append(path: Path, row: dict) -> None:
    """Append one row to progress.jsonl atomically (a single write of one line),
    so a watcher can tail -f the file across re-runs without corruption."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _write_summary(run_dir: Path, traces: list[dict], meta: dict, mode: str) -> None:
    """Aggregate the per-question traces into summary.json / summary.md."""
    if not traces:
        return
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "n": 0})
    abst = {"ok": 0, "n": 0}
    for t in traces:
        s = t.get("summary") or {}
        if not s:
            continue
        if s.get("abstention"):
            abst["n"] += 1
            abst["ok"] += int(s.get("score", 0))
        else:
            by_task[s["task"]]["n"] += 1
            by_task[s["task"]]["ok"] += int(s.get("score", 0))
    tot_ok = sum(v["ok"] for v in by_task.values())
    tot_n = sum(v["n"] for v in by_task.values())
    summary = {
        "run_at": meta["run_at"], "llm": meta["llm"], "judge": meta["judge"],
        "reranker": meta["reranker"], "embedding": meta["embedding"],
        "mode": mode, "git": meta.get("git", ""),
        "tasks": {k: dict(v) for k, v in sorted(by_task.items())},
        "abstention": dict(abst),
        "answerable_total": {"ok": tot_ok, "n": tot_n},
        "questions": {t["meta"]["question_id"]: t.get("summary")
                      for t in traces if t.get("summary")},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    n_ = max(1, tot_n)
    lines = ["# LongMemEval-S x personos — run summary", "",
             f"- run: {meta['run_at']} - mode {mode} - git `{meta.get('git', '')}`",
             f"- benchmark LLM: {meta['llm']} - judge: {meta['judge']}",
             f"- reranker: {meta['reranker']} - embedding: {meta['embedding']}", "",
             f"## Answerable questions: **{tot_ok}/{tot_n} "
             f"({100 * tot_ok / n_:.1f}%)**", "",
             "| task | correct/total | accuracy |", "|---|---|---|"]
    for k, v in sorted(by_task.items()):
        lines.append(f"| {k} | {v['ok']}/{v['n']} | {100 * v['ok'] / max(1, v['n']):.1f}% |")
    if abst["n"]:
        lines += ["", f"## Abstention questions (unanswerable, judge asks 'did you identify "
                     f"it as unanswerable?'): **{abst['ok']}/{abst['n']} "
                     f"({100 * abst['ok'] / max(1, abst['n']):.1f}%)**"]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(prog="python -m scripts.bench.longmemeval")
    ap.add_argument("--data", default=str(LME_PATH),
                    help="path to longmemeval_s_cleaned.json (default data/longmemeval_s_cleaned.json; "
                         "--download fills it)")
    ap.add_argument("--download", action="store_true",
                    help="download longmemeval_s_cleaned.json from Hugging Face and exit")
    ap.add_argument("--all", action="store_true",
                    help="run every question in the dataset (default: --questions-file or first 5)")
    ap.add_argument("--n-sessions", type=int, default=999,
                    help="how many haystack sessions to load per question (default 999 = all); "
                         "questions whose answer_session is beyond the cap are skipped")
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "deep"])
    ap.add_argument("--concurrency", type=int, default=1,
                    help="how many questions run in parallel inside this process "
                         "(default 1; each question has its own user_id, so they cannot "
                         "contaminate each other's store)")
    ap.add_argument("--questions-file", default="",
                    help="a JSONL file with one question_id per line — the question_ids this "
                         "process should run. Use this to split a 500-question run across "
                         "processes / days (e.g. head -250 vs tail -250 of --all).")
    ap.add_argument("--run-dir", default="",
                    help="artifact directory (default data/bench/runs/<timestamp>). Pass the "
                         "same --run-dir across re-launches to accumulate traces and append "
                         "to the same progress.jsonl.")
    ap.add_argument("--force-reingest", action="store_true",
                    help="delete the user's store and re-ingest the haystack from scratch "
                         "(default: reuse existing atoms and only re-run the answer+judge)")
    ap.add_argument("--force-reanswer", action="store_true",
                    help="re-answer and re-judge even when a trace JSON already exists "
                         "(default: skip questions whose trace is already on disk)")
    args = ap.parse_args()

    if args.download:
        download_dataset(Path(args.data))
        return
    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found: {args.data} — run with --download first")

    run_dir = Path(args.run_dir) if args.run_dir else (
        Path("data/bench/runs") / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"

    cfg = get_config()
    llm = build_provider("llm", cfg.llm_provider)
    if cfg.judge_api_key:
        from dataclasses import replace
        jcfg = replace(cfg,
                       llm_base_url=cfg.judge_base_url or cfg.llm_base_url,
                       llm_api_key=cfg.judge_api_key,
                       llm_model=cfg.judge_model or cfg.llm_model)
        judge_llm = build_provider("llm", cfg.judge_provider or "openai", cfg=jcfg)
        judge_desc = f"{cfg.judge_model or cfg.llm_model} (LME official yes/no, verbatim from " \
                     f"xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py)"
    else:
        judge_llm = llm
        judge_desc = f"{cfg.llm_model} (LME official yes/no, verbatim from " \
                     f"xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py)"
    embedder = build_provider("embedder", cfg.embedder_provider)
    rp = cfg.reranker_provider or ("openai" if cfg.rerank_api_key and cfg.rerank_model else "noop")
    scorer = None if rp == "noop" else build_provider("reranker", rp)
    reranker = ScoringReranker(scorer) if scorer else NoopReranker()
    db = Database()

    meta = {"mode": args.mode,
            "llm": cfg.llm_model,
            "embedding": cfg.embedding_model,
            "reranker": (f"{rp}:{cfg.rerank_model}" if scorer else "off (fusion order)"),
            "judge": judge_desc,
            "answerer": "no second stage — the R5 answer is the prediction (official LME protocol)",
            "run_at": time.strftime("%Y-%m-%d %H:%M"), "data": args.data,
            "dataset": "longmemeval_s_cleaned.json (xiaowu0162/longmemeval-cleaned, MIT)"}
    try:
        import subprocess
        meta["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    instances = load_dataset(args.data)
    if args.questions_file:
        wanted = set(
            line.strip() for line in Path(args.questions_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        wanted_insts = [x for x in instances if x.question_id in wanted]
        miss = len(wanted) - len(wanted_insts)
        if miss:
            print(f"warning: {miss}/{len(wanted)} question_ids in {args.questions_file} "
                  f"are not in the dataset (typo or wrong dataset file)")
    elif args.all:
        wanted_insts = instances
    else:
        wanted_insts = instances[:5]
    print(f"run dir: {run_dir}\ndataset: {args.data} ({len(instances)} questions, "
          f"{sum(1 for x in instances if x.is_abstention)} abstention)\n"
          f"benchmark LLM: {cfg.llm_model} - judge: {judge_desc} - reranker: {meta['reranker']}\n"
          f"questions to run this launch: {len(wanted_insts)}")

    # Filter by n-sessions gating (analogous to LoCoMo's pick_answerable) —
    # if the answer_session_id is beyond the cap, the question is unfair and
    # skipped; reported in the summary as "skipped_unreachable".
    eligible = []
    skipped_unreachable = 0
    for inst in wanted_insts:
        if inst.is_abstention or not inst.answer_session_ids:
            eligible.append(inst)
        elif inst.answer_reachable(args.n_sessions):
            eligible.append(inst)
        else:
            skipped_unreachable += 1
    if skipped_unreachable:
        print(f"n-sessions cap={args.n_sessions}: {skipped_unreachable} questions skipped "
              f"(answer_session beyond the cap)")

    conc = max(1, args.concurrency)
    print(f"per-process concurrency: {conc}", flush=True)

    traces: list[dict] = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(run_instance, inst, args, run_dir, llm, judge_llm,
                          embedder, reranker, db, meta): inst for inst in eligible}
        for fut in as_completed(futs):
            inst = futs[fut]
            try:
                t = fut.result()
            except Exception:   # noqa: BLE001
                logger.exception(f"instance {inst.question_id} crashed")
                t = None
            if t is not None:
                traces.append(t)
                s = t.get("summary") or {}
                _progress_append(progress_path, {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "question_id": inst.question_id, "task": inst.question_type,
                    "abstention": inst.is_abstention,
                    "score": int(s.get("score", 0)),
                    "secs": s.get("secs", 0),
                    "status": "ok",
                })

    # -- Re-aggregate from disk so this summary reflects ALL completed traces
    #    on this run dir, not just the ones this process produced. Re-runs
    #    (yesterday's shards + today's shards) end up in one summary. --
    all_traces = []
    for jf in sorted(run_dir.glob("*.trace.json")):
        try:
            all_traces.append(json.loads(jf.read_text(encoding="utf-8")))
        except Exception:   # noqa: BLE001
            logger.warning(f"could not parse {jf}, ignoring")
    _write_summary(run_dir, all_traces, meta, args.mode)

    if all_traces:
        scored = [t for t in all_traces if t.get("summary")]
        tot_ok = sum(t["summary"]["score"] for t in scored
                     if not t["summary"].get("abstention"))
        tot_n = sum(1 for t in scored if not t["summary"].get("abstention"))
        abst_ok = sum(t["summary"]["score"] for t in scored
                      if t["summary"].get("abstention"))
        abst_n = sum(1 for t in scored if t["summary"].get("abstention"))
        print(f"\n===== run dir total: answerable {tot_ok}/{tot_n} "
              f"({100 * tot_ok / max(1, tot_n):.1f}%) - abstention {abst_ok}/{abst_n} "
              f"({100 * abst_ok / max(1, abst_n):.1f}%) =====")
        print(f"summary: {run_dir/'summary.md'}")
        print(f"progress log: {progress_path}")

    embedder.close()
    if scorer:
        scorer.close()
    if judge_llm is not llm:
        judge_llm.close()
    llm.close()
    db.close()


if __name__ == "__main__":
    main()