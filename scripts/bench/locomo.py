"""LoCoMo-10 x personos benchmark, self-contained: download -> ingest -> answer
-> judge -> trace/report on disk.

Flow: take the first N sessions of a conversation -> feed them message by
message through SessionWriter (W0 evidence / W1 boundaries / W2 cell building,
with now_dt set to the session time) -> select the questions whose evidence
lies entirely within the loaded sessions -> run the whole fast path in
run_recall (R0 scope -> R1 dual retrieval -> R2 reranking -> R5 draft ->
R3' review) -> produce an English answer with the mode-A answerer -> score it
with a Mem0-style binary judge. Everything talks to the core directly; no HTTP
server is involved.

The benchmark protocol reports two scoring conventions:
- the **Mem0 convention** (comparable with published numbers): the answerer is
  mode A (brief = the R5 answer; memories = the atoms hit in the top 20
  reranked cells, dated and grouped by cell, aligned with what R5 saw, under
  the trust-the-brief rule), and the judge scores the answerer's output;
- the **product convention** (what the product actually returns): no second
  stage, and the judge scores the R5 answer directly (judge_r5).
State the protocol difference whenever these numbers are compared externally.

Models come from the environment: the memory system uses the service
``PERSONOS_LLM_*`` / ``PERSONOS_EMBEDDING_*`` / ``PERSONOS_RERANK_*`` settings,
the judge uses ``PERSONOS_JUDGE_*`` (falling back to the service LLM), so the
model being benchmarked and the model scoring it can differ.

Artifacts (git-ignored, under data/bench/runs/<run_id>/):
- <conv>.trace.json — full pipeline trace: ingest boundaries, closed cells,
  per-question rewrite/pool/ranked/review/deep/memories, gold-chain break point
- <conv>.report.md — the per-conversation final report with error analysis
- summary.json / summary.md — the run-level aggregate across conversations

Usage:
  python -m scripts.bench.locomo --download                      # fetch locomo10.json
  python -m scripts.bench.locomo --conv conv-26 --n-sessions 3 --n-questions 8   # smoke
  python -m scripts.bench.locomo --all --n-sessions 99 --n-questions 999 \
      --concurrency 4                                          # the full benchmark
"""

from __future__ import annotations

import argparse
import json
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
from personos.online.profile_consolidate import run_user_consolidation
from personos.online.profile_render import render as render_profile
from personos.storage.profile_store import ProfileStore
from personos.online.write_path import SessionWriter
from personos.providers.registry import build as build_provider
from personos.storage.atom_store import AtomStore
from personos.storage.cell_store import CellStore
from personos.storage.db import Database
from personos.storage.evidence_store import EvidenceStore
from scripts.bench.judge import Verdict, answer_mode_a, judge

LOCOMO_PATH = Path("data/locomo10.json")   # --download fills this; --data overrides
# The dataset's canonical home is the snap-research GitHub repo; there is no
# official Hugging Face upload, so a community mirror is tried first and the
# GitHub raw file is the fallback. Both are validated structurally after the
# download (ten conversations), never trusted blind.
HF_REPO = "KimmoZZZ/locomo"
HF_FILE = "locomo10.json"
GITHUB_RAW = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Truncation limit for long text in the trace (the complete verbatim text lives
# in the store and is never cut).
_CAP_RAW = 2000

# How wide the answerer's memory view is: the atoms hit in the top N reranked
# cells, the same convention the public recall API uses for supporting memories.
_ANSWER_MEM_CELLS = 20    # aligned with what R5 sees (10 made the two layers disagree)

# category -> question-type name, matching Mem0's official evaluation code
# (memory-benchmarks/locomo/prompts.py) and the original locomo repository's
# task_eval/evaluation.py.
_CAT_NAME = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop",
             5: "adversarial (no gold answer)"}


# ═══════════════════════════ data adapter ═══════════════════════════════
# Pure parsing; no LLM or storage calls. Image captions (img_url +
# blip_caption, generated offline by BLIP as part of the dataset) are folded
# into the text as "[shared an image: ...]", so image semantics travel through
# the text channel into evidence, extraction and retrieval alike.

_DT_FORMATS = ["%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"]   # "1:56 pm on 8 May, 2023"
_DIA_RE = re.compile(r"^D(\d+):(\d+)$")


@dataclass
class Exchange:
    """One utterance (consecutive turns by the same speaker merged). Both LoCoMo
    participants are human, and each is fed in as an independent message.

    holder: speaker_a -> "user" (the first-person subject, the owner of the
    memory); speaker_b -> their real name (a human participant, not an
    assistant). Deliberately not routed through the assistant_reply channel:
    epistemically that channel is "assistant context" (a suggestion or guess
    only becomes fact once the user endorses it), whereas LoCoMo's speaker_b is
    a real person whose self-reported facts should be fully extractable,
    attributable, and whose verbatim words belong in the evidence pool.
    """
    holder: str
    text: str
    dia: str = ""


@dataclass
class LocomoSession:
    idx: int                       # 1-based session number (matches the D# in dia_id)
    dt: datetime                   # when the session happened (the now_dt baseline)
    exchanges: list[Exchange] = field(default_factory=list)


@dataclass
class LocomoQA:
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)   # ["D1:3", ...]
    category: int = 0                                    # 1-5; 5 = adversarial (dropped by convention)

    def evidence_sessions(self) -> set[int]:
        out = set()
        for e in self.evidence:
            m = _DIA_RE.match(e.strip())
            if m:
                out.add(int(m.group(1)))
        return out


@dataclass
class LocomoConversation:
    sample_id: str
    speaker_a: str                 # mapped to user
    speaker_b: str                 # mapped to their real name
    sessions: list[LocomoSession] = field(default_factory=list)
    qa: list[LocomoQA] = field(default_factory=list)


def _parse_dt(raw: str) -> datetime:
    s = (raw or "").strip()
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    raise ValueError(f"cannot parse LoCoMo session timestamp: {raw!r}")


def _group_turns(turns: list[dict], speaker_a: str, speaker_b: str) -> list[Exchange]:
    """Merge consecutive turns by the same speaker into one utterance, using the
    speaker's **real name** as the holder (e.g. "Caroline" / "Melanie").

    EvidenceRecord.holder is a free-form string, so passing the real name
    straight through makes the write path render the dialogue line as
    "Melanie: ..." (the model can see the attribution), and the extracted atoms
    carry the real name as their holder too.
    speaker_a, the first-person viewpoint, is always "user" - they own the
    memory. speaker_b is a human participant and does not go through the
    assistant channel, which is epistemically "assistant context" and would
    down-weight extraction.
    """
    out: list[Exchange] = []
    for t in turns or []:
        spk, text, dia = t["speaker"], (t.get("text") or "").strip(), t.get("dia_id", "")
        # Image messages: the caption is inlined into the text. This matches the
        # EverMemOS evaluation convention - image understanding already happened
        # upstream in BLIP, and the memory layer only ever consumes text.
        # One turn stays one line, which keeps the line count unchanged (the
        # trace zips split("\n") against dia to align gold evidence).
        if t.get("img_url"):
            text = (f"[shared an image: {t.get('blip_caption', 'an image')}] {text}").strip()
        holder = "user" if spk == speaker_a else spk
        if out and out[-1].holder == holder:
            last = out[-1]
            last.text = (last.text + "\n" + text).strip()
            last.dia = (last.dia + "," + dia).lstrip(",")
        else:
            out.append(Exchange(holder=holder, text=text, dia=dia))
    return out


def load_conversations(path: str | Path) -> list[LocomoConversation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[LocomoConversation] = []
    for c in raw:
        conv_d = c["conversation"]
        speaker_a, speaker_b = conv_d["speaker_a"], conv_d["speaker_b"]
        lc = LocomoConversation(sample_id=c["sample_id"], speaker_a=speaker_a, speaker_b=speaker_b)
        sids = sorted(
            (k for k in conv_d if re.fullmatch(r"session_\d+", k)),
            key=lambda s: int(s.split("_")[1]),
        )
        for s in sids:
            idx = int(s.split("_")[1])
            lc.sessions.append(LocomoSession(
                idx=idx,
                dt=_parse_dt(conv_d[f"{s}_date_time"]),
                exchanges=_group_turns(conv_d[s], speaker_a, speaker_b),
            ))
        for qa in c.get("qa", []):
            lc.qa.append(LocomoQA(
                question=qa.get("question", ""),
                answer=qa.get("answer", ""),
                evidence=[str(e) for e in (qa.get("evidence") or [])],
                category=int(qa.get("category", 0) or 0),
            ))
        out.append(lc)
    return out


def pick_answerable(lc: LocomoConversation, session_idxs: set[int],
                    *, exclude_categories=(5,)) -> list[LocomoQA]:
    """Select questions whose evidence lies entirely within the loaded sessions
    and whose category is not excluded, so that everything asked is in the store."""
    return [
        qa for qa in lc.qa
        if qa.category not in exclude_categories and qa.evidence and qa.evidence_sessions() <= session_idxs
    ]


def download_dataset(dest: Path = LOCOMO_PATH) -> Path:
    """Fetch locomo10.json: the Hugging Face mirror first, the official
    snap-research GitHub raw file as the fallback. huggingface_hub is imported
    lazily and kept out of the package dependencies on purpose."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
        shutil.copy(src, dest)
        origin = f"huggingface:{HF_REPO}"
    except ImportError:
        origin = ""
    except Exception as e:  # mirror missing/moved — fall through to the canonical source
        print(f"huggingface mirror unavailable ({type(e).__name__}), trying GitHub...")
        origin = ""
    if not origin:
        import httpx
        dest.write_bytes(httpx.get(GITHUB_RAW, follow_redirects=True, timeout=120).content)
        origin = "github:snap-research/locomo"
    convs = json.loads(dest.read_text(encoding="utf-8"))
    assert isinstance(convs, list) and len(convs) == 10 and all(
        "conversation" in c and "qa" in c for c in convs), \
        f"{dest} does not look like locomo10.json — refusing to benchmark against it"
    print(f"dataset downloaded from {origin}: {dest} ({dest.stat().st_size / 1e6:.1f} MB, "
          f"{len(convs)} conversations)")
    return dest


# ═══════════════════════════ harness ════════════════════════════════════

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
                "forced_close": r.forced_close,      # the turn-count safety valve, not an LLM decision
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
    deep path. The verbatim support for each memory is filled in by run_conv,
    which has the ev_store.

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
    ingest record exists, the exact dia -> evidence_id mapping is used instead.
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


def _error_kind(q: dict) -> str:
    """A first-pass attribution for a wrong answer (heuristic; break_point and
    human review are the final word)."""
    ans = (q.get("answer") or "").strip()
    if ans.lower().startswith("i don't have"):
        verdict = (q.get("review") or {}).get("verdict")
        if verdict == "insufficient_material":
            return ("refusal - review found insufficient material (check extraction: was the "
                    "information extracted at all? then check retrieval)")
        return ("refusal - review returned a defect and it still refused (check whether R5 "
                "answering or the answerer is too conservative)")
    return "answer disagrees with the gold (check ranking -> R5 materials -> answerer -> judge)"


# ═══════════════════════════ markdown report ════════════════════════════

def _t(x) -> str:
    return str(x).replace("\n", " ").replace("|", "\\|")


def render_md(trace: dict, out_path: str | Path) -> Path:
    """The per-conversation final report in markdown: totals under both
    conventions, a per-category breakdown, review verdicts and an error analysis."""
    m = trace.get("meta", {})
    s = trace.get("summary", {})
    qa = trace.get("qa", [])
    st = trace.get("store", {})
    lines = [f"# LoCoMo x personos final benchmark report - {_t(m.get('conv'))}", ""]
    lines.append(f"- run: {_t(m.get('run_at', ''))} - mode {_t(m.get('mode'))} - "
                 f"LLM / judge {_t(m.get('llm'))} / {_t(m.get('judge'))} - git `{_t(m.get('git', ''))}`")
    lines.append(f"- answerer: {_t(m.get('answerer', ''))}")
    lines.append(f"- scope: the first {m.get('n_sessions')} sessions - {s.get('n_questions')} "
                 f"valid questions (cat5 adversarial questions excluded)")
    lines.append(f"- final store state: {len(st.get('cells', []))} cells / "
                 f"{len(st.get('atoms', []))} atoms / {len(st.get('evidence', []))} evidence records")
    n = max(1, len(qa))
    lines.append(f"- average {sum(q.get('secs', 0) for q in qa)/n:.0f}s per question")
    lines.append("")
    lines.append(f"## Total: **Mem0 convention {s.get('score')}/{s.get('n_questions')} "
                 f"({100*s.get('score', 0)/n:.1f}%) - product convention (R5 judged directly) "
                 f"{s.get('score_r5', 0)}/{s.get('n_questions')} "
                 f"({100*s.get('score_r5', 0)/n:.1f}%)**")
    lines.append("")
    lines.append("| category | correct/total (Mem0 convention) | accuracy |")
    lines.append("|---|---|---|")
    for k, v in sorted(s.get("by_category", {}).items()):
        lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {v[0]}/{v[1]} | {100*v[0]/max(1,v[1]):.0f}% |")
    vs = s.get("verdicts", {})
    lines.append("")
    lines.append(f"review verdicts: ok {vs.get('ok', 0)} - answer_defect {vs.get('answer_defect', 0)}"
                 f" ({s.get('retried', 0)} re-answers) - insufficient_material "
                 f"{vs.get('insufficient_material', 0)} - escalated to the deep path on "
                 f"{s.get('escalated', 0)} questions")
    lines.append("")
    lines.append("## Error analysis")
    wrong = [q for q in qa if not q.get("judge")]
    if not wrong:
        lines.append("No wrong answers.")
    for i, q in enumerate(wrong, 1):
        lines.append(f"### E{i} [{_CAT_NAME.get(q['category'], q['category'])}] {_t(q['question'])}")
        lines.append(f"- gold: **{_t(q['gold'])}** - evidence: {', '.join(q.get('evidence', []))}")
        lines.append(f"- break point: **{_t(q.get('break_point') or _error_kind(q))}**")
        for c in (q.get("gold_chain") or []):
            marks = (f"evidence {'y' if c['ev_id'] else 'n'} -> atoms "
                     f"{'y' if c['n_extracted'] else 'n'} ({c['n_extracted']})"
                     f" -> retrieval top10 {'y' if c['in_hits'] else 'n'}"
                     f" -> materials {'y' if c['in_material'] else 'n'}"
                     f" -> cited by R5 {'y' if c['cited'] else 'n'}")
            lines.append(f"  - `{_t(c['dia'])}` {marks}")
        rv = q.get("review") or {}
        lines.append(f"- review: {_t(rv.get('verdict'))}{' (re-answered)' if rv.get('retried') else ''} - "
                     f"critique: {_t((rv.get('critique') or '')[:120])}")
        lines.append(f"- memories: {'; '.join(_t(m2['text'])[:80] for m2 in q.get('memories', [])[:3]) or '(none)'}")
        lines.append(f"- answer: {_t((q.get('answer') or '')[:150])} - judge: {_t(q.get('judge_raw'))}"
                     f" - R5 judged directly: {'correct' if q.get('judge_r5') else 'wrong'} "
                     f"{_t(q.get('judge_r5_raw'))}")
        lines.append("")
    lines.append("> The break point is the first stage at which the gold evidence chain broke; "
                 "each question's full pipeline is in the trace JSON in the same directory.")
    p = Path(out_path)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ═══════════════════════════ per-conversation run ═══════════════════════

def run_conv(lc: LocomoConversation, args, run_dir: Path, llm, judge_llm,
             embedder, reranker, db: Database, meta: dict) -> dict:
    """Benchmark one conversation end to end; returns its trace dict (also
    written to disk as <conv>.trace.json / <conv>.report.md)."""
    print(f"\n===== conversation {lc.sample_id}: {len(lc.sessions)} sessions / "
          f"{len(lc.qa)} questions; taking the first {args.n_sessions} sessions =====")

    # Each conversation gets its own user namespace, and a re-run clears that
    # user's tables first (idempotent, so atom counts do not double).
    uid = f"locomo-{lc.sample_id}"
    have = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
    if args.skip_ingest:
        assert have > 0, f"--skip-ingest needs user data to already be loaded: {uid}"
    elif have:
        for t in ("evidence", "atoms", "atom_chains", "memcells", "session_context"):
            db.execute(f"DELETE FROM {t} WHERE user_id=%s", (uid,))
        # The profile is derived from those same tables, so clearing them
        # without clearing profile_versions would let a stale snapshot leak into
        # the next run.
        db.execute("DELETE FROM profile_versions WHERE user_id=%s", (uid,))
    ev_store, atoms = EvidenceStore(db, uid), AtomStore(db, uid)
    cells = CellStore(db, uid)
    profiles = ProfileStore(db, uid)

    trace: dict = {"meta": {**meta, "conv": lc.sample_id, "n_sessions": args.n_sessions}}

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
        print(f"load complete ({time.time()-t0:.0f}s):")
        for s in trace["ingest"]:
            print(f"  s{s['session']} @{s['dt'][:10]}: {len(s['exchanges'])} exchanges -> "
                  f"{len(s['closed_cells'])} cells / {s['atoms_total']} atoms total ({s['secs']}s)")

    # -- Optionally build the per-user profile once, before answering. Today is
    #    pinned to the last loaded session so any relative phrasing in the
    #    profile stays temporally aligned with the question time. --
    profile_full = profile_traits = ""
    if args.with_profile:
        if profiles.current() is None or not args.skip_ingest:
            t1 = time.time()
            new_version = run_user_consolidation(
                llm, cells_store=cells, atoms_store=atoms,
                profile_store=profiles, today=last_dt.date())
            print(f"profile consolidated in {time.time()-t1:.0f}s"
                  + (f" (v{new_version})" if new_version else " (no-op: nothing to consolidate)"))
        ver = profiles.current()
        if ver is not None:
            profile_full = render_profile(ver.profile, mode="full")
            profile_traits = render_profile(ver.profile, mode="traits")
            print(f"profile v{ver.version} ready "
                  f"({len(profile_full)} chars full / {len(profile_traits)} chars traits)")
        else:
            print("profile: empty (no cells to summarize)")
        trace["meta"]["profile"] = (
            f"on (v{ver.version})" if ver is not None else "on (empty)")
    else:
        trace["meta"]["profile"] = "off"

    # Load only: write a load trace for inspection and stop before answering, so
    # a batch can fill several conversations first and answer them together.
    if args.ingest_only:
        n_atoms = db.fetch_one("SELECT COUNT(*) AS n FROM atoms WHERE user_id=%s", (uid,))["n"]
        n_cells = db.fetch_one("SELECT COUNT(*) AS n FROM memcells WHERE user_id=%s", (uid,))["n"]
        print(f"[ingest-only] {lc.sample_id} loaded: user={uid} atoms={n_atoms} cells={n_cells}")
        out = run_dir / f"{lc.sample_id}.ingest.json"
        out.write_text(json.dumps({"meta": trace["meta"], "ingest": trace["ingest"],
                                   "atoms": n_atoms, "cells": n_cells,
                                   "profile_chars": len(profile_full)},
                                  ensure_ascii=False, indent=1))
        print(f"load trace written to {out}")
        return trace

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
    print(f"{len(answerable)} answerable questions (all evidence within the first "
          f"{args.n_sessions} sessions, cat5 excluded); answering {len(qas)} of them")

    # -- Per question: the whole fast path -> answerer -> judge, N at a time --
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
                               now_dt=last_dt, mode=args.mode, reranker=reranker,
                               profile_full=profile_full, profile_traits=profile_traits)
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
                v = judge(judge_llm, question=qa.question, gold=qa.answer, prediction=answer)
                try:
                    v5 = judge(judge_llm, question=qa.question, gold=qa.answer,
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
                print(f"[{lc.sample_id} {i}/{len(qas)}] {'ok' if v.ok else 'X'}/"
                      f"{'ok' if v5.ok else 'X'} cat{qa.category} ({rec['secs']}s, "
                      f"review={verdict}{deep_part}) Q: {qa.question[:60]}", flush=True)
                return rec
            except Exception:   # noqa: BLE001
                logger.exception(f"Q{i} failed, attempt {attempt}/3")
                if attempt < 3:
                    time.sleep(60)
        print(f"[{lc.sample_id} {i}/{len(qas)}] X/X cat{qa.category} (failed after 3 retries) "
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
    print(f"answering concurrency: {conc}", flush=True)
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

    trace_path = run_dir / f"{lc.sample_id}.trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1))
    n = max(1, len(qa_records))
    print(f"\n[{lc.sample_id}] result: Mem0 convention {n_ok}/{len(qa_records)} "
          f"({100*n_ok/n:.0f}%) - product convention (R5 judged directly) "
          f"{n_ok5}/{len(qa_records)} ({100*n_ok5/n:.0f}%)")
    for cat, (ok, tot) in sorted(by_cat.items()):
        print(f"  cat{cat}: {ok}/{tot}")
    print(f"review verdicts: {dict(verdicts)}")
    md_path = render_md(trace, run_dir / f"{lc.sample_id}.report.md")
    print(f"trace: {trace_path}\nreport: {md_path}")
    return trace


# ═══════════════════════════ entry ══════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(prog="python -m scripts.bench.locomo")
    ap.add_argument("--conv", action="append", default=[],
                    help="conversation id (repeatable); default conv-26, or all ten with --all")
    ap.add_argument("--all", action="store_true", help="all ten conversations")
    ap.add_argument("--n-sessions", type=int, default=3)
    ap.add_argument("--n-questions", type=int, default=8)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "deep"])
    ap.add_argument("--data", default=str(LOCOMO_PATH),
                    help="path to locomo10.json (default data/locomo10.json; --download fills it)")
    ap.add_argument("--download", action="store_true",
                    help="download locomo10.json from Hugging Face and exit")
    ap.add_argument("--run-dir", default="",
                    help="artifact directory (default data/bench/runs/<timestamp>)")
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
    ap.add_argument("--with-profile", action="store_true",
                    help="consolidate the per-user profile before answering and feed it into "
                         "R0 rewrite + the deep track (profile_full) and R5 drafting (profile_traits). "
                         "Off by default to match Mem0-protocol comparability.")
    args = ap.parse_args()

    if args.download:
        download_dataset(Path(args.data))
        return
    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found: {args.data} — run with --download first")

    run_dir = Path(args.run_dir) if args.run_dir else (
        Path("data/bench/runs") / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    # The benchmark LLM (W1/W2/R0/R3/R5 + the answerer) and the judge are
    # deliberately separable: scoring with the model being measured is a
    # conflict of interest, so PERSONOS_JUDGE_* can point elsewhere.
    llm = build_provider("llm", cfg.llm_provider)
    if cfg.judge_api_key:
        from dataclasses import replace
        jcfg = replace(cfg,
                       llm_base_url=cfg.judge_base_url or cfg.llm_base_url,
                       llm_api_key=cfg.judge_api_key,
                       llm_model=cfg.judge_model or cfg.llm_model)
        judge_llm = build_provider("llm", cfg.judge_provider or "openai", cfg=jcfg)
        judge_desc = f"{cfg.judge_model or cfg.llm_model} (Mem0-style binary, v2 with relative-time conversion)"
    else:
        judge_llm = llm
        judge_desc = f"{cfg.llm_model} (Mem0-style binary, v2 with relative-time conversion)"
    embedder = build_provider("embedder", cfg.embedder_provider)
    rp = cfg.reranker_provider or ("openai" if cfg.rerank_api_key and cfg.rerank_model else "noop")
    scorer = None if rp == "noop" else build_provider("reranker", rp)
    # R2: wrap the scorer so a failed rerank degrades to pass-through order,
    # same as the product recall path.
    reranker = ScoringReranker(scorer) if scorer else NoopReranker()
    db = Database()

    meta = {"mode": args.mode,
            "llm": cfg.llm_model,
            "embedding": cfg.embedding_model,
            "reranker": (f"{rp}:{cfg.rerank_model}" if scorer else "off (fusion order)"),
            "judge": judge_desc,
            "answerer": "mode A, two conventions (brief = the R5 answer; memories = "
                        "atoms hit in the top 20 reranked cells, grouped by cell, "
                        "under trust-the-brief; judge_r5 scores R5 directly)",
            "chain": "step 1 write + step 2 fast path",
            "profile": "on" if args.with_profile else "off (Mem0 protocol default)",
            "run_at": time.strftime("%Y-%m-%d %H:%M"), "data": args.data}
    try:   # the short git hash goes into meta so a report traces back to a code version
        import subprocess
        meta["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    convs = load_conversations(args.data)
    wanted = convs if args.all else [c for c in convs if c.sample_id in (args.conv or ["conv-26"])]
    if not wanted:
        raise SystemExit(f"no such conversation: {args.conv}")
    print(f"run dir: {run_dir}\nbenchmark LLM: {cfg.llm_model} - judge: {judge_desc} - "
          f"reranker: {meta['reranker']}")

    traces = [run_conv(lc, args, run_dir, llm, judge_llm, embedder, reranker, db, meta)
              for lc in wanted]

    # -- Run-level aggregate across the conversations answered in this run --
    if not args.ingest_only:
        scored = [t for t in traces if t.get("summary")]
        tot_ok = sum(t["summary"]["score"] for t in scored)
        tot_ok5 = sum(t["summary"]["score_r5"] for t in scored)
        tot_q = sum(t["summary"]["n_questions"] for t in scored)
        by_cat: dict[str, list] = defaultdict(lambda: [0, 0])
        for t in scored:
            for k, (ok, n_) in t["summary"]["by_category"].items():
                by_cat[k][0] += ok
                by_cat[k][1] += n_
        summary = {"run_at": meta["run_at"], "llm": meta["llm"], "judge": meta["judge"],
                   "reranker": meta["reranker"], "embedding": meta["embedding"],
                   "mode": args.mode, "git": meta.get("git", ""),
                   "score": tot_ok, "score_r5": tot_ok5, "n_questions": tot_q,
                   "by_category": {k: v for k, v in sorted(by_cat.items())},
                   "conversations": {t["meta"]["conv"]: t["summary"] for t in scored}}
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        n_ = max(1, tot_q)
        lines = [f"# LoCoMo-10 x personos — run summary", "",
                 f"- run: {meta['run_at']} - mode {args.mode} - git `{meta.get('git', '')}`",
                 f"- benchmark LLM: {meta['llm']} - judge: {meta['judge']}",
                 f"- reranker: {meta['reranker']} - embedding: {meta['embedding']}", "",
                 f"## Total: **Mem0 convention {tot_ok}/{tot_q} ({100*tot_ok/n_:.1f}%) - "
                 f"product convention {tot_ok5}/{tot_q} ({100*tot_ok5/n_:.1f}%)**", "",
                 "| conversation | Mem0 | product (R5) |", "|---|---|---|"]
        for t in scored:
            s = t["summary"]
            lines.append(f"| {t['meta']['conv']} | {s['score']}/{s['n_questions']} "
                         f"({100*s['score']/max(1,s['n_questions']):.0f}%) | "
                         f"{s['score_r5']}/{s['n_questions']} "
                         f"({100*s['score_r5']/max(1,s['n_questions']):.0f}%) |")
        lines += ["", "| category | correct/total | accuracy |", "|---|---|---|"]
        for k, (ok, n2) in sorted(by_cat.items()):
            lines.append(f"| cat{k} {_CAT_NAME.get(int(k), '')} | {ok}/{n2} | {100*ok/max(1,n2):.0f}% |")
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n===== run total: Mem0 convention {tot_ok}/{tot_q} ({100*tot_ok/n_:.1f}%) - "
              f"product convention {tot_ok5}/{tot_q} ({100*tot_ok5/n_:.1f}%) =====")
        print(f"summary: {run_dir/'summary.md'}")

    embedder.close()
    if scorer:
        scorer.close()
    if judge_llm is not llm:
        judge_llm.close()
    llm.close()
    db.close()


if __name__ == "__main__":
    main()
