"""Recall orchestration (fused architecture §5, decisions 1 and 2): the fast path runs end to end,
R0 -> R1 (atom pool) -> unit assembly (chain weaving) -> R2 -> R5 -> R3' adjudication; the deep track
branches off.

The public service API calls it and renders a clean view; benchmark and scenario scripts call it
directly too. Dependencies are passed in explicitly (no coupling to rt or FastAPI), which keeps it
testable and reusable.

The branches: mode=deep runs only R0 (to give the deep track the resolved query, date window and
domains) and then goes straight to the deep-track agent (its own benchmarking scope); mode=auto runs
the whole fast path — R5 produces a draft first, R3' adjudicates "draft + the same materials": an
answer defect means one re-answer with the critique attached, and if it is still defective or the
material is insufficient it escalates to the deep track (escalated=True), whose final answer
overrides the fast-path answer.
A deep-track crash does not drag the main path down: it falls back to the fast-path answer and
`escalated` still faithfully records what happened.

Benchmark breakdown: one summary INFO line per question (recall done mode=... R0=..s R1=..s
units=..s R2=..s R5=..s ...), with each station logging its own INFO/WARNING as well; grouping them
by trace id is enough to locate "which stage went wrong".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from .arbitrate import ReviewResult, review_answer
from .chain_face import UnitAssembly, assemble_units
from .rerank import NoopReranker, Reranker, rerank_cells
from .retrieval import (
    AtomHit, CellHit, MemoryAnswer, QueryRewrite, answer_from_cells, rewrite_query, search_atoms,
)
from .session_context import build_history
from .visual_query import VisualRewrite, enrich_query_with_image
from ..storage.atom_store import AtomStore
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.evidence_store import EvidenceStore

# Three public modes: auto (escalates automatically) | fast (fast path only) | deep (straight to the
# deep track, for standalone benchmarking)
PUBLIC_MODES = ("auto", "fast", "deep")


def _fmt_day(dt) -> str:
    """Format a date, accepting a datetime, an ISO string, or None. The time_start/time_end of a
    rewrite may be a string (carried straight out of the LLM's JSON), and formatting that as a
    datetime used to raise ValueError and turn the whole recall into a 502."""
    if not dt:
        return ""
    if isinstance(dt, str):
        return dt[:10]                      # for an ISO string take the date part; truncating a non-ISO string is harmless too
    try:
        return f"{dt:%Y-%m-%d}"
    except (ValueError, TypeError):
        return str(dt)[:10]


def _no_answer_note(rw, review, deep) -> str:
    """An objective account when there is no answer: what was searched / the conclusion / the likely
    reason.

    The honesty obligation of a memory framework: when it cannot answer it gives neither a plausible-
    sounding conclusion nor a silent empty-handed return — it states the search scope and the gap, so
    the caller can rephrase the question or add to the memory. Assembled deterministically, with no
    LLM call.
    """
    parts = []
    if rw:
        scope = []
        if rw.subject:
            scope.append(f"subject {rw.subject!r}")
        tw = f"{_fmt_day(rw.time_start)}~{_fmt_day(rw.time_end)}".strip("~")
        if tw:
            scope.append(f"time window {tw}")
        if rw.domains:
            scope.append("domains " + "/".join(rw.domains))
        parts.append("Searched: " + (", ".join(scope) if scope else "the whole store") + ". ")
    if deep:
        parts.append("The deep track read through memory over several steps and did not "
                     "find anything that answers this within its step budget. ")
    else:
        parts.append("No memory relevant enough to answer this was found. ")
    reason = ("Either this never came up in a recorded conversation, or it did but was "
              "not indexed for retrieval.")
    if review and review.critique:
        reason += f" Gap: {review.critique}"
    parts.append("Possible reason: " + reason)
    return "".join(parts)


@dataclass
class RecallOutcome:
    """The raw product of one recall, handed to each API layer to render its own view."""
    query: str
    mode: str
    rw: QueryRewrite | None = None      # the R0 five-tuple (the no-answer note cites the search scope; even mode=deep runs R0 first)
    hits: list[AtomHit] = field(default_factory=list)      # the R1 atom pool (in two-way RRF fusion order)
    ranked: list[CellHit] = field(default_factory=list)    # the material units after R2 rerank (plain + woven)
    draft: MemoryAnswer | None = None        # the first R5 draft (the original answer, before adjudication or re-answering)
    reviews: list[ReviewResult] = field(default_factory=list)   # adjudication history (0 to 2 entries)
    retried: bool = False                    # re-answered once after an answer_defect verdict
    ans: MemoryAnswer | None = None          # the final answer (the ok draft, the re-answer, or the deep-track answer when auto escalated)
    escalated: bool = False                  # auto mode, and adjudication still found a defect or insufficient material -> escalated to the deep track
    deep: "DeepOutcome | None" = None          # the deep-track product (trace, number written back; None when the deep track did not run)
    asm: UnitAssembly | None = None          # unit-assembly inspection (pool / chain / woven / plain counts + the boundary note)
    vis: VisualRewrite | None = None       # visual rewrite (only when the caller sent an image; None = a pure text recall)
    secs: dict[str, float] = field(default_factory=dict)   # per-station cost (for the summary log and benchmark statistics)
    # Capabilities that were asked for but unavailable, stated rather than
    # silently skipped: an image that could not be looked at, a deep track that
    # could not run. The answer is still returned; the caller decides what to
    # do about the gap.
    warnings: list[str] = field(default_factory=list)

    def to_public(self, *, atoms=None, evidence=None, media_store=None,
                  max_memories: int = 20) -> dict:
        """Flatten to a plain dict, for an HTTP response or a simpler caller.

        This is the *only* renderer. The server calls it too, so the library
        and the API cannot drift into describing the same recall differently.

        What is deliberately not here: scores, prompts, raw model output, deep
        track steps. They are internals; publishing them turns implementation
        detail into contract.
        """
        verdict = self.reviews[-1].verdict if self.reviews else None
        insufficient = verdict == "insufficient_material"

        mem: list[dict] = []
        if atoms is not None and evidence is not None and not insufficient:
            from .views import memory_view

            picked: list = []
            if self.deep and self.ans and self.ans.cited_cells:
                for cid in self.ans.cited_cells:
                    picked.extend(atoms.list_by_cell(cid))
            elif verdict is not None:
                for hit in self.ranked[:max_memories]:
                    picked.extend(a.atom for a in hit.atoms)
            mem = [memory_view(a, evidence, media_store)
                   for a in picked[:max_memories] if a is not None]

        out = {
            "query": self.query,
            "mode": self.mode,
            "verdict": verdict,
            "critique": self.reviews[-1].critique if self.reviews else "",
            "retried": self.retried,
            "escalated": self.escalated,
            "answer": self.ans.answer if self.ans else "",
            "cited_cells": self.ans.cited_cells if self.ans else [],
            "memories": mem,
        }
        if self.warnings:
            out["warnings"] = list(self.warnings)
        # Only when an image was actually sent. Without this a caller cannot
        # tell "recognised the person but could not answer" from "did not
        # recognise anyone" — and the advice to give the user differs
        # completely (rephrase the question vs send a clearer photo).
        if self.vis is not None:
            out["visual"] = {
                "faces": self.vis.faces,
                "matched": [{"character_id": m.get("character_id", ""),
                             "name": m.get("name", "")} for m in self.vis.matched],
                "resolved_query": self.vis.query,
            }
        return out


def run_recall(
    llm, embedder, atoms: AtomStore, cells: CellStore, evidence: EvidenceStore,
    *,
    session_id: str,
    query: str,
    now_dt: datetime,
    mode: str = "auto",
    top_k: int = 30,
    rewrite: bool = True,
    reranker: Reranker | None = None,
    media_store=None, mllm=None,   # image-viewing dependencies (passed through to the deep track; without them the deep track reads content_inline instead of looking at images)
    image: bytes | None = None, image_content_type: str = "image/jpeg",   # an image the caller sent along with the question
    visual_deps=None,      # visual-rewrite dependencies (rt.visual_deps); only triggers when given together with `image`, otherwise this is a pure text path
    profile_full: str = "", profile_traits: str = "",   # profile injection (full -> R0 and the deep track; traits -> R5); empty = no profile, behaving exactly as before
    scenario: str = "",   # the caller's scenario description (may be empty) -> R0 / R5 / deep track; empty keeps the default path byte-for-byte identical
) -> RecallOutcome:
    """The full fast-path orchestration: R0 preprocessing -> R1 two-way atom retrieval (associative
    and domain RRF, a top_k pool) -> unit assembly (dedup atoms by chain, weave >=2-node chains into
    memcell', map single-node and free-floating atoms to plain cells) -> R2 rerank (one rerank over
    all units) -> R5 draft -> R3' adjudication (draft + the same materials). mode=deep goes straight
    to the deep track; in auto mode an answer defect is re-answered once, and a remaining defect or
    insufficient material escalates to the deep track, which overrides the final answer."""
    out = RecallOutcome(query=query, mode=mode)
    t0 = time.perf_counter()

    def mark(stage: str):
        out.secs[stage] = round(time.perf_counter() - t0, 3)   # cumulative anchors: each stage's cost is a difference between two of them

    # The basis for reference resolution (the rolling summary + the last few turns). Passing `cells`
    # folds video segments into their episode instead of 20+ lines of verbatim dialogue.
    history = build_history(evidence, session_id, llm=llm, cell_store=cells)
    mark("hist")

    # Visual understanding before R0: write the people and the scene in the image into the query, so
    # the purely textual path that follows can use the visual information too.
    # It only triggers when the caller sent an image; on failure it always falls back to the original
    # query (see visual_query), leaving pure-text recall byte-for-byte unchanged.
    q0 = query
    if image is not None and visual_deps is not None:
        out.vis = enrich_query_with_image(
            visual_deps, query=query, image=image, content_type=image_content_type,
            history=history, scenario=scenario, now_dt=now_dt)
        q0 = out.vis.query
    mark("vis")

    out.rw = (rewrite_query(llm, raw_query=q0, history=history, now_dt=now_dt,
                            profile=profile_full, scenario=scenario)
              if rewrite else QueryRewrite(original=q0, resolved=q0))
    mark("R0")

    def run_deep_safe(**kw):
        """Deep track wrapper: a crash here falls back to the fast answer,
        and ``escalated`` still records that escalation was attempted.

        The import is deferred on purpose. The deep track is an agent built on
        langchain, which is a large dependency that most users of a memory
        library do not want; deferring it keeps `pip install personos` small
        and lets the feature be an extra. A missing dependency is reported as
        such rather than being swallowed as "the deep track crashed", which
        would be true but useless.
        """
        try:
            from .deep_recall import run_deep
        except ImportError:
            from ..errors import deep_track_skipped

            note = deep_track_skipped()
            if note not in out.warnings:
                out.warnings.append(note)
            logger.warning(note)
            return None
        try:
            return run_deep(llm, embedder, atoms, cells, evidence, reranker=reranker,
                            media_store=media_store, mllm=mllm, profile=profile_full,
                            scenario=scenario, **kw)
        except Exception as e:   # noqa: BLE001
            logger.exception(f"deep track failed, falling back to the fast answer "
                             f"q={query!r}: {e}")
            return None

    if mode == "deep":
        # Straight to the deep track (its own benchmarking scope): R0's resolved query, date window
        # and domains go into the handoff package, and every fast-path station is skipped
        out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw)
        out.ans = out.deep.ans if out.deep else MemoryAnswer(answer="")
        mark("deep")
    else:
        pool = search_atoms(embedder, atoms, rewrite=out.rw, top_n=top_k)
        out.hits = pool.atoms
        mark("R1")
        # Unit assembly (§5.2): pool atoms -> dedup by chain -> woven memcell' / plain units + the
        # generic boundary note (D-C10)
        out.asm = assemble_units(out.hits, ChainStore(atoms.db, atoms.user_id), cells,
                                 llm, query=out.rw.resolved, beyond=pool.beyond)
        mark("units")
        materials = out.asm.units
        if not materials:
            # Empty-retrieval short circuit (§5 decision 1): no answering, no adjudication, just
            # return empty; auto escalates to the deep track
            out.ans = MemoryAnswer(answer="")
            mark("R5")
            if mode == "auto":
                out.escalated = True
                out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw,
                                         review=None, fast_hits=[])
                if out.deep and out.deep.ans.answer:
                    out.ans = out.deep.ans   # the deep-track final answer overrides; an empty answer does not
                mark("deep")
        else:
            # R2 rerank: plain cells and woven memcell's are structurally identical, so one rerank
            # covers both (it is a temporary view, with no special-case logic); everything downstream
            # (R5, R3', the re-answer, the deep-track handoff) consumes the reranked order —
            # otherwise a real reranker would have run for nothing
            out.ranked = rerank_cells(reranker or NoopReranker(), out.rw.resolved, materials)
            materials = out.ranked
            mark("R2")
            boundary = out.asm.boundary
            # R5 answers first (the draft), then adjudication checks "draft + the same materials"
            # claim by claim (the reordering from §5 decision 1)
            out.draft = answer_from_cells(llm, query=out.rw.resolved, subject=out.rw.subject,
                                          hits=materials, now_dt=now_dt, boundary=boundary,
                                          profile=profile_traits, scenario=scenario)
            out.ans = out.draft
            mark("R5")
            out.reviews.append(review_answer(llm, query=query, draft=out.draft, hits=materials,
                                             resolved=out.rw.resolved, subject=out.rw.subject,
                                             boundary=boundary))
            mark("R3")
            # Two-tier disposition (§5 decision 2): an answer defect means one re-answer with the
            # critique attached; insufficient material goes straight to the deep track
            if out.reviews[-1].verdict == "answer_defect":
                out.ans = answer_from_cells(llm, query=out.rw.resolved, subject=out.rw.subject,
                                            hits=materials, now_dt=now_dt, boundary=boundary,
                                            feedback=out.reviews[-1].critique, profile=profile_traits,
                                            scenario=scenario)
                out.retried = True
                mark("R5r")
                out.reviews.append(review_answer(
                    llm, query=query, draft=out.ans, hits=materials,
                    resolved=out.rw.resolved, subject=out.rw.subject, boundary=boundary))
                mark("R3r")
            verdict = out.reviews[-1].verdict if out.reviews else "ok"
            if mode == "auto" and verdict != "ok":
                out.escalated = True
                out.deep = run_deep_safe(query=query, now_dt=now_dt, rw=out.rw,
                                         review=out.reviews[-1] if out.reviews else None,
                                         fast_hits=materials)
                if out.deep and out.deep.ans.answer:
                    out.ans = out.deep.ans      # the deep-track final answer overrides; an empty answer does not (the fast-path wording is kept)
                mark("deep")

    stages = [s for s in ("hist", "R0", "R1", "units", "R2", "R5", "R3", "R5r", "R3r", "deep")
              if s in out.secs]
    parts = " ".join(f"{b}={out.secs[b] - out.secs[a]:.1f}s" for a, b in zip(stages, stages[1:]))
    deep_part = (f" deep_steps={len(out.deep.steps)} remembered={out.deep.remembered}"
                 if out.deep else "")
    unit_part = (f" units={len(out.asm.units)}/{out.asm.n_chains}ch/{out.asm.n_woven}wv"
                 if out.asm else "")
    logger.info(f"recall finished mode={mode}{' escalated' if out.escalated else ''}{deep_part} "
                f"{' retried' if out.retried else ''} {parts} pool={len(out.hits)}{unit_part} "
                f"verdict={out.reviews[-1].verdict if out.reviews else '-'} "
                f"cited={len(out.ans.cited_cells) if out.ans else 0} "
                f"ans_chars={len(out.ans.answer) if out.ans else 0} q={query!r}")
    # No-answer fallback: an empty answer becomes an objective account (what was searched, the
    # conclusion, the reason) rather than a silent empty-handed return
    if out.ans is None or not out.ans.answer.strip():
        out.ans = MemoryAnswer(answer=_no_answer_note(out.rw, out.reviews[-1] if out.reviews else None,
                                                      out.deep))
    # The full picture of the recall: final answer + cited cells + the reranked material units
    # (topic and the text of the atoms that hit), for troubleshooting
    ranked_detail = "\n".join(
        f"    [{i}] score={h.rerank_score if h.rerank_score is not None else h.score:.3f} "
        f"topic={h.cell.topic!r} atoms={[a.atom.text for a in h.atoms]}"
        for i, h in enumerate(out.ranked, 1)) if out.ranked else "    (none)"
    logger.info(f"recall result q={query!r} cited={out.ans.cited_cells}\n"
                f"  -- final answer --\n{out.ans.answer}\n"
                f"  -- reranked material units(n={len(out.ranked)}) --\n{ranked_detail}")
    return out
