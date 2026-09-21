"""Fast-path recall (fused architecture §3): R0 preprocessing -> R1 two-route atom retrieval + RRF ->
R5 answering.

The retrieval unit is the atom (two routes, associative and domain; the topic route has been
removed); the material unit is the memcell (either a plain cell or the temporary memcell' view woven
from a chain — structurally identical and invisible downstream). Atoms never enter any rerank or
answering material (a retrieval unit must not double as a reference answer).
Writes are never reconciled — a duplicate is a redundant index, and conflicts are consumed at
answering time (D1).
The content of an open (unclosed) segment is in no vector pool; the current conversation is answered
from the session context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
from loguru import logger

from ..models import (
    DOMAIN_VOCAB, MemCell, MemoryAtom, atom_anchor, ensure_aware, now, vocab_menu,
)
from ..storage.atom_store import AtomStore
from .llm import ChatLLM, chat_json, with_scenario

# Directives for caller-scenario injection (they only tune attention and level of detail; facts are
# never altered, invented, or omitted) -- see with_scenario
_SCEN_DIR_REWRITE = ("When the question is ambiguous, bias domain guesses and expansion terms toward "
                     "the caller's subject area; never override the literal question.")
_SCEN_DIR_ANSWER = ("Use it only to shape emphasis and level of detail in the answer; it never changes "
                    "which facts are true, nor licenses stating anything absent from the materials.")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


# -- Hit structures: the atom is the retrieval unit (R1), the memcell is the material / rerank unit
# (from R2 onwards) --

@dataclass
class AtomHit:
    atom: MemoryAtom
    similarity: float        # this atom's max cosine over the query faces (best of the associative and domain routes)
    rrf: float = 0.0         # RRF fusion score (a ranking quantity; the deep track's single-route retrieval does not use it and leaves it at 0)


@dataclass
class CellHit:
    """A material unit: either a plain memcell or the memcell' woven from a chain (a temporary view
    that is completely invisible downstream).

    For a memcell': cell.id = the chain id, topic = the chain title, episode = the woven text,
    t_start/t_end = the span of its member cells, covers = the full set of member cell ids — `covers`
    is only used for the short/long handle mapping of citations (mN -> member cells) and never enters
    any prompt text.
    A plain unit has an empty `covers`, and citation expansion treats it as [its own cell.id].
    """

    cell: MemCell
    score: float                     # the RRF fusion score of the unit's best atom (a ranking quantity, not a similarity)
    best_sim: float                  # the similarity of the unit's best atom (explainable: how close the meaning is)
    atoms: list[AtomHit] = field(default_factory=list)   # the pool atoms that hit this unit (descending similarity)
    rerank_score: float | None = None                   # the R2 rerank score (None when rerank did not run)
    covers: list[str] = field(default_factory=list)     # for a woven unit, the full set of member cell ids; empty means [its own cell.id]


def _cosine(qmat: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """qmat: (f,d) query faces x mat: (n,d) documents -> an (n,f) cosine matrix. Safe for zero
    vectors."""
    if mat.size == 0:
        return np.zeros((0, qmat.shape[0]), dtype=np.float32)
    qn = qmat / (np.linalg.norm(qmat, axis=1, keepdims=True) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mn @ qn.T


def _maxsim_ranking(
    query_vecs: list[np.ndarray], pool: list[tuple[MemoryAtom, np.ndarray]],
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    """MaxSim: take each atom's max cosine over the query faces, then aggregate by memcell_id taking
    the max -> a cell ranking.

    A cell only contributes its strongest atom (which diversifies the candidate pool automatically and
    stops one rich cell from flooding it).
    Returns (the cell ranking [descending similarity], cell -> MaxSim score, cell -> atom -> its best
    face similarity).
    """
    if not pool:
        return [], {}, {}
    qmat = np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs])
    sims = _cosine(qmat, np.stack([v for (_, v) in pool])).max(axis=1)   # each atom's best face
    cell_scores: dict[str, float] = {}
    atom_sims: dict[str, dict[str, float]] = {}
    for (atom, _), s in zip(pool, sims):
        atom_sims.setdefault(atom.memcell_id, {})[atom.id] = float(s)
        s = float(s)
        if s > cell_scores.get(atom.memcell_id, -1.0):
            cell_scores[atom.memcell_id] = s
    ranking = sorted(cell_scores, key=lambda c: cell_scores[c], reverse=True)
    return ranking, cell_scores, atom_sims


def _vec_ranking(query_vecs: list[np.ndarray], entries: list[tuple[MemCell, np.ndarray]]) -> list[str]:
    """A pool with one vector per cell (e.g. topic vectors): take the max cosine over the query faces
    -> a cell id ranking (descending)."""
    if not entries:
        return []
    sims = _cosine(np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs]),
                   np.stack([v for (_, v) in entries])).max(axis=1)
    order = sorted(range(len(entries)), key=lambda i: sims[i], reverse=True)
    return [entries[i][0].id for i in order]


_RRF_K = 60   # the RRF smoothing constant (fixed by fused architecture §3; a larger k flattens the differences between ranks)


def _rrf(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Multiple rankings -> a fusion score: the sum of 1/(k+rank), with rank starting at 1.

    Items several routes agree on float up while single-route noise is diluted; an item that appears
    in only one route accumulates only that route's contribution.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


# -- R0 - query preprocessing: one lightweight LLM call produces all five fields (qtype has been
# split out, D-C1) --
_REWRITE_SYS = (
    "# Role\n"
    "You are the query preprocessor of a memory retrieval system: turn the user's CURRENT QUESTION into "
    "the five retrieval fields below.\n\n"
    "# Input\n"
    "Current time (the anchor for converting relative times), Recent dialogue (the basis for resolving "
    "references), Current question.\n\n"
    "# Output (strict JSON, all five fields at once)\n"
    "1. resolved: apply exactly TWO minimal edits to the question and change nothing else —\n"
    "   (a) resolve references/ellipsis: replace 'it / that / afterwards / this matter' with the actual "
    "referent from the Recent dialogue; no reference word may remain in resolved;\n"
    "   (b) relative → absolute time: convert 'last week / a few days ago / yesterday / last year' into "
    "absolute dates using the Current time (e.g. if now is 2026-08-25, 'yesterday' → '2026-08-24').\n"
    "   Do not stuff dates into a question that carries no time meaning; apart from these two edits keep "
    "the original wording — no rephrasing, no added interpretation.\n"
    "2. subject: whose affairs the question asks about — 'what have I been busy with' → 'user'; "
    "'how is Caroline's exhibition going' → 'Caroline'. Not about a specific person, or unclear → null.\n"
    "3. expansions: 0-5 retrieval-boosting terms or paraphrases, in the SAME language as the question "
    "('leg pain' → 'sports; injury'; 'busy with which project' → 'preparing which event; any plans'). "
    "When the subject is a proper name, include name-carrying paraphrases of the asked relation "
    "('what movie did Joanna watch' → 'Joanna's movies; film Joanna saw'; 'what did Nate do for "
    "Joanna' → \"Nate's gift to Joanna; Nate's favor for Joanna\") — memories may phrase the fact in "
    "first person ('user') or third person, and a name-carrying face bridges the wording gap. "
    "Be restrained: no wild leaps, no far-fetched associations; when unsure, give fewer or [].\n"
    "4. time_start / time_end: when the question carries its own time constraint, convert it into an "
    "absolute-date window (ISO yyyy-MM-dd) using the Current time — 'what did we talk about last week' → "
    "start = that Monday, end = that Sunday. No time constraint → both null.\n"
    "5. domains: which life domains the ANSWER will live in (not which domain the question's surface "
    "belongs to), 0-3, Dxx codes only. Vocabulary:\n  "
    + vocab_menu(DOMAIN_VOCAB) + "\n"
    "   Memories are filed by domain — a correct guess lets the domain route hit fast. Guess when you "
    "reasonably can (1-3 codes); don't stay empty out of over-caution, and don't force clearly "
    "irrelevant domains. [] only when there is truly no basis.\n\n"
    "# Output format\n"
    'JSON only, no extra text: {"resolved":"the full question after reference resolution and '
    'relative-to-absolute time conversion","subject":"user|name|null","expansions":["reasonable '
    'terms"],"time_start":"2026-08-11|null","time_end":"2026-08-17|null","domains":["D05"]}'
)


@dataclass
class QueryRewrite:
    original: str
    resolved: str               # the question after reference resolution and relative-to-absolute time conversion (used by both retrieval and answering)
    subject: str = ""           # the subject of the question (the anchor for R5's attribution check; "user" or a person's name; empty = could not be determined)
    expansions: list[str] = field(default_factory=list)   # expansion terms (the query faces of the associative route)
    time_start: str = ""        # the absolute date window (ISO yyyy-MM-dd; empty = unbounded; feeds the deep-track tools' start/end_date directly)
    time_end: str = ""
    domains: list[str] = field(default_factory=list)      # the domain route's anchors (the D axis)
    system: str = ""
    user: str = ""
    raw: str = ""


def _iso_date(v) -> str:
    """LLM output -> a normalized ISO date string (yyyy-MM-dd); null or a parse failure -> an empty
    string (that side of the window is unbounded)."""
    d = ensure_aware(v)
    return d.date().isoformat() if d else ""


def rewrite_query(llm: ChatLLM, *, raw_query: str, history: list[tuple[str, str]] | None = None,
                  now_dt: datetime | None = None, profile: str = "", scenario: str = "") -> QueryRewrite:
    """R0: one LLM call produces all five fields (reference resolution, question subject, date window,
    expansion terms, domain guess); a parse failure degrades to the original query (fault-tolerant,
    never blocking).

    profile: the user profile text block (may be empty) — injected to help with reference and
    ellipsis resolution, subject determination, and converting the user's habitual time expressions.
    scenario: the caller's scenario description (may be empty) — biases the domain guess and the
    expansion terms toward the caller's subject area.
    """
    now_dt = now_dt or now()
    hist = "\n".join(f"{h}: {t}" for h, t in (history or [])) or "(no history)"
    prof = f"\n\n{profile}" if profile else ""
    user = (f"Current time: {now_dt.isoformat()} (the anchor for converting relative times in the "
            f"dialogue into absolute dates)\n\nRecent dialogue:\n{hist}{prof}\n\nCurrent question:\n{raw_query}")
    sys = with_scenario(_REWRITE_SYS, "# Input", scenario, _SCEN_DIR_REWRITE)
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    try:
        obj, raw = chat_json(llm, messages, max_tokens=800, temperature=0.2, stage="rewrite_query")
        resolved = str(obj.get("resolved") or "").strip() or raw_query
        subject = str(obj.get("subject") or "").strip()
        if subject.lower() in ("null", "none", "无"):
            subject = ""
        exp = [s for s in (str(x).strip() for x in (obj.get("expansions") or obj.get("associations") or []))
               if s][:5]
        doms = [d for d in (str(x).strip() for x in (obj.get("domains") or [])) if d in DOMAIN_VOCAB]
        ts, te = _iso_date(obj.get("time_start")), _iso_date(obj.get("time_end"))
        logger.info(f"R0 rewrite subject={subject or '-'} exp={exp} "
                    f"window={ts or '-'}/{te or '-'} domains={doms or '[]'} "
                    f"resolved={resolved!r}")
        return QueryRewrite(original=raw_query, resolved=resolved, subject=subject,
                            expansions=exp, time_start=ts, time_end=te, domains=doms,
                            system=sys, user=user, raw=raw)
    except Exception as e:
        logger.warning(f"R0 rewrite parse failed, falling back to the original query: {e}")
        return QueryRewrite(original=raw_query, resolved=raw_query,
                            system=sys, user=user,
                            raw=getattr(e, "raw", "") or str(e))


# -- R1 - two-route recall: the associative route (divergent, whole store) in parallel with the domain
# route (convergent, the subset inside the guessed domains) -> atom-level RRF --

_PER_CELL_CAP = 10   # max atoms from one cell in the pool (a guard on atom-level pool selection, so a rich cell cannot flood it and squeeze others out)


def _date_window(start_date: str, end_date: str) -> tuple[datetime | None, datetime | None]:
    """A yyyy-MM-dd string -> (lower bound, upper bound + 1 day); the date window covers WHOLE DAYS
    INCLUSIVE of both endpoints. A malformed value makes that side unbounded.

    The fast path (which does no filtering) and the deep-track tools (search_atoms / find_cells) share
    this one definition of window semantics, so the two cannot drift apart.
    """
    lo = ensure_aware(start_date) if start_date else None
    hi = ensure_aware(end_date) if end_date else None
    if hi:
        hi = hi + timedelta(days=1)
    return lo, hi


def _in_window(anchor: datetime | None, lo: datetime | None, hi: datetime | None) -> bool:
    """Whether the anchor falls inside the window; under a time filter an object with no anchor is
    invisible (if we do not know when it held, it cannot pass itself off as in force)."""
    return not ((lo and (anchor is None or anchor < lo)) or (hi and (anchor is None or anchor >= hi)))


def _atom_maxsim(query_vecs: list[np.ndarray],
                 rows: list[tuple[MemoryAtom, np.ndarray]]) -> dict[str, float]:
    """Each atom's max cosine over the query faces -> {atom_id: sim} (the raw material for atom-level
    ranking, with no cell aggregation)."""
    if not rows:
        return {}
    qmat = np.stack([np.asarray(v, dtype=np.float32) for v in query_vecs])
    sims = _cosine(qmat, np.stack([v for (_, v) in rows])).max(axis=1)
    return {a.id: float(s) for (a, _), s in zip(rows, sims)}


@dataclass
class AtomPool:
    """The R1 product: the atom pool in RRF fusion order + the candidates beyond it (used by the
    boundary note to name their chain titles).

    `beyond` only collects ranks after the pool filled up — an atom squeezed out by the per-cell cap
    still has its fact in the episode of a shown cell, so it does not count as missing material.
    """
    atoms: list[AtomHit] = field(default_factory=list)
    beyond: list[AtomHit] = field(default_factory=list)


def search_atoms(
    embedder: Embedder, atom_store: AtomStore,
    *,
    rewrite: QueryRewrite,
    top_n: int = 30,
    per_cell_cap: int = _PER_CELL_CAP,
    start_date: str = "",
    end_date: str = "",
    holder: str = "",
    domains_filter: list[str] | None = None,
) -> AtomPool:
    """Two-route atom retrieval -> RRF (k=60) fusion -> pool selection, taking top_n atoms (D-C6: the
    recall unit is the atom).

    - Associative route: the resolved query plus each expansion term are **embedded separately** into
      several query faces; each atom takes its max cosine over those faces, against the whole atom
      pool — recall first, so nothing is missed.
    - Domain route: the resolved query as-is with no expansion, against only the subset of atoms in
      the guessed domains — precision first, to avoid false hits (an atom in the right domain whose
      wording did not match semantically can still be found). With no domain guess this route is
      empty and RRF naturally degenerates into a single route.
    - Each route contributes its top 2 x top_n ranks to RRF (oversampling), and top_n is taken after
      fusion.
    - Pool selection: at most per_cell_cap atoms per cell; once the pool is full the remaining ranks
      go to `beyond` (used by the chain face's boundary note).
    - start/end_date / holder / domains_filter: the structured HARD FILTERS of the deep-track tools
      (the fast path leaves them all empty); the date window is applied to the atom anchor
      (occurrence_time, falling back to recorded_at) over whole inclusive days.
      Note that domains_filter is not rewrite.domains: the former is a hard filter (a condition the
      agent set explicitly), while the latter is the domain route's anchor (a recall hint R0 guessed,
      which filters out no atom at all).
    """
    lo, hi = _date_window(start_date, end_date)
    rows = []
    for a, v in atom_store.all_with_embeddings():
        if not a.memcell_id:
            continue
        if (lo or hi) and not _in_window(atom_anchor(a), lo, hi):
            continue
        if holder and a.holder != holder:
            continue
        if domains_filter and not (set(a.domains) & set(domains_filter)):
            continue
        rows.append((a, v))

    # The associative route's query faces (face[0] is always the resolved query, which the domain
    # route reuses, so it is embedded only once)
    faces = [rewrite.resolved] + list(rewrite.expansions)
    qvecs = list(embedder.embed(faces))
    assoc_sims = _atom_maxsim(qvecs, rows)
    assoc_ranking = sorted(assoc_sims, key=lambda aid: -assoc_sims[aid])

    dom_sims: dict[str, float] = {}
    if rewrite.domains and rows:
        dset = set(rewrite.domains)
        dom_sims = _atom_maxsim([qvecs[0]], [(a, v) for a, v in rows if dset & set(a.domains)])
    dom_ranking = sorted(dom_sims, key=lambda aid: -dom_sims[aid])

    route_depth = 2 * top_n                                # per-route oversampling depth (60 = 2 x 30)
    fused = _rrf([assoc_ranking[:route_depth], dom_ranking[:route_depth]])
    best_sim = {aid: max(assoc_sims.get(aid, 0.0), dom_sims.get(aid, 0.0)) for aid in fused}
    order = sorted(fused, key=lambda aid: (-fused[aid], -best_sim[aid]))   # ties are broken by similarity

    atom_map = {a.id: a for a, _ in rows}
    per_cell: dict[str, int] = {}
    pool = AtomPool()
    for aid in order:
        atom = atom_map.get(aid)
        if atom is None:
            continue                                       # defensive: every key in `fused` comes from `rows`, so this is unreachable in theory
        if len(pool.atoms) < top_n:
            n = per_cell.get(atom.memcell_id, 0)
            if n < per_cell_cap:
                per_cell[atom.memcell_id] = n + 1
                pool.atoms.append(AtomHit(atom=atom, similarity=best_sim[aid], rrf=fused[aid]))
            # Squeezed out by the per-cell cap: that cell's fact is already in the shown episode, so
            # it does not count as missing material -> skip it (it does not go into `beyond`)
        else:
            pool.beyond.append(AtomHit(atom=atom, similarity=best_sim[aid], rrf=fused[aid]))
    pool_detail = "\n".join(
        f"    [{i}] sim={h.similarity:.3f} rrf={h.rrf:.4f} {h.atom.text!r}"
        for i, h in enumerate(pool.atoms, 1))
    logger.info(f"R1 search_atoms rows={len(rows)} faces={len(faces)} domains={rewrite.domains or '[]'} "
                f"assoc={len(assoc_ranking)} domain_route={len(dom_ranking)} pool={len(pool.atoms)} "
                f"beyond={len(pool.beyond)} q={rewrite.resolved!r}\n"
                f"  matched atom pool(n={len(pool.atoms)}):\n{pool_detail}")
    return pool


# -- R5 - answering: the episode as the primary material plus the atoms that hit, in one LLM call --

# The conflict-consumption rule (where D1 lands): the answering and adjudication prompts share this
# one wording, so restating it in several places cannot drift.
_MOST_RECENT_RULE = (
    "A fact stated several times and never mentioned again afterwards: the MOST RECENT statement "
    "is the current state — even if older statements are more frequent or more detailed."
)
CONFLICT_RULE = (
    "Conflict consumption: when multiple cells in the materials state different things about the same "
    "fact (one says 'loves apples', another says 'allergic to apples'), order them by the cell time "
    "window and treat the later one as the new state; present the evolution ('earlier ... later ...'); "
    "an explicit correction always wins. " + _MOST_RECENT_RULE + " Never average, never pick one "
    "arbitrarily."
)

# R5 v2 (2026-09-09, from error attribution: the root cause of the 68 questions in the refusal bucket
# was that, given a flat list of rules, the model takes the path of least effort — it writes a soft
# refusal that sounds well-reasoned, and adjudication lets a draft with no claims through because
# there is nothing to check it against). So the flat rule list became a CoT procedure: refusing is
# made expensive and has to leave a trace (declaring block by block), which gives R3' something
# checkable to grab onto. All the hard-rule semantics are preserved (conflict, attribution, fidelity,
# degree, citations).
_ANSWER_SYS = (
    "# Role\n"
    "You are the answerer of a personal memory system: answer the USER QUESTION strictly from the given "
    "MEMORY MATERIALS.\n\n"
    "# How to read the materials\n"
    "Each material block is one topic unit, opened by a \"━━━ mN ━━━\" separator line; the next "
    "line is a header with the dialogue time and topic, followed by the segment narrative "
    "(episode) — the ONLY primary material. Times, attribution and qualifiers are already "
    "written into the narrative itself.\n\n"
    "# Working procedure (follow IN ORDER, then answer)\n"
    "STEP 1 — Scan: read EVERY block m1..mN and collect every fact relevant to the question; do not "
    "stop at the first hit — one overlooked block is a wrong answer.\n"
    "STEP 2 — Connect: link facts across blocks when the question needs it (who did what, where, "
    "why); the materials may state pieces in separate blocks — join them.\n"
    "STEP 3 — Infer when needed: you MAY draw a direct, single-step inference supported by the "
    "materials ('planned for 2026-09' → 'has not happened yet; it is a plan'; 'runs the shop "
    "personally, handling everything' → a small operation). Do NOT dismiss a reasonable inference as "
    "mere speculation; only avoid chaining several speculative leaps.\n"
    "STEP 4 — Time: use only two anchors — the dual-time format inside the materials "
    "('last week (2026-08-18)') and the Current time. Convert 'how long ago / is it still ...' "
    "against the Current time. Never do calendar arithmetic from memory, never use times from "
    "outside the materials, never guess a date the materials do not state.\n"
    "STEP 5 — Verify (list/count questions): re-scan every block for any matching item you missed, "
    "then count the distinct items the materials actually contain and verify your list has exactly "
    "that many. For COUNT questions ('how many ...'): list the distinct instances first, then count "
    "them; MERGE duplicates — the same event mentioned in several blocks counts ONCE — and exclude "
    "plans/intentions that never happened. Give the exact count, never 'at least N'.\n"
    "STEP 6 — Answer, observing these hard rules:\n"
    "- " + CONFLICT_RULE + "\n"
    "- Detail fidelity: full names/numbers/frequencies exactly as written in the materials — no "
    "generalization ('3 pizzas' stays '3 pizzas'; 'hot yoga' stays 'hot yoga', never 'exercise').\n"
    "- Degree questions: for qualitative/how-much questions where the materials give indirect but "
    "real evidence, answer at the degree the evidence supports ('shows real interest — brought it "
    "up twice herself') and state the evidential boundary; do not refuse the whole question just "
    "because the evidence is indirect, and do not inflate the degree beyond the evidence.\n"
    "- Attribution check: when the answer's subject is NOT the QUESTION SUBJECT, say 'that was said/"
    "done by X' and keep the attribution straight — never pass one person's matter off as the asked "
    "subject's. If the asked subject has nothing in the materials, say so honestly; never substitute "
    "someone else's.\n"
    "- Citations: besides answer, list the cell handles you relied on in cells (m1, m2 ...); the "
    "answer body itself must not contain handles.\n\n"
    "# Refusal is a last resort\n"
    "If the materials give ANY direct, indirect, or single-step-inferable evidence — even hedged — "
    "you MUST answer (state the evidential basis if indirect). If you can DESCRIBE the thing but "
    "hesitate to name it, commit to the specific name — describing-but-refusing-to-name counts as a "
    "refusal and is not allowed. Only when the materials contain genuinely nothing usable may you "
    "say the information is not found — and then the answer MUST state that you checked every block "
    "m1..mN, name the closest block (by handle) and what exactly it lacks; never a bare 'I don't "
    "know', never a plausible-sounding guess dressed as fact.\n\n"
    "# Requirements\n"
    "- Answer in the language of the question.\n"
    "- Style: a standard factual statement — third person, neutral, conclusion first, stating the facts "
    "directly; no conversational tone (never 'you told me / I remember / I guess'); the tone does not "
    "change with how the question is phrased. This is the memory service's factual output — the caller "
    "(an external agent) composes the conversation.\n"
    "- Include time cues (absolute dates). If the materials hold nothing relevant, say so honestly — no "
    "fabrication, no guessing.\n\n"
    "# Output (JSON only)\n"
    '{"answer":"...","cells":["m1"]}'
)


@dataclass
class MemoryAnswer:
    answer: str                 # the direct answer to the question (an empty string means memory holds nothing relevant)
    cited_cells: list[str] = field(default_factory=list)   # the cell ids cited (rule 6)
    system: str = ""
    user: str = ""
    raw: str = ""


def _date_str(dt: datetime | None) -> str:
    """A date -> a bare 'yyyy-MM-dd' (used for the cell time window; no brackets are wrapped around it
    here); no date -> an empty string."""
    dt = ensure_aware(dt)
    return f"{dt:%Y-%m-%d}" if dt else ""


def cell_lead(c: MemCell) -> str:
    """Assemble one language-neutral material header from metadata programmatically: the dialogue
    time (an evidence timestamp, so no LLM has to recognize it) + the topic.

    Shared by the R2 rerank document and the R3/R5 material blocks, which keeps the unit as seen by
    all three stations identical.
    The format is fixed English (the metadata frame does not change with the language of the stored
    content; the language of the material body comes from the episode and topic themselves).
    """
    ts, te = _date_str(c.t_start), _date_str(c.t_end)
    if ts and te:
        when = ts if ts == te else f"{ts} to {te}"
    else:
        when = "date unknown"
    return f"[dialogue {when} | topic: {c.topic or '(no topic)'}]"


def cell_block(hit: CellHit, handle: str) -> str:
    """One material unit: a "━━━ handle ━━━" separator line + the metadata header (time + topic) +
    the episode (the only primary material).

    A plain cell and a woven memcell' render identically (a temporary view, invisible downstream):
    a memcell' has topic = the chain title, time = the span of its member cells, episode = the woven
    text. `covers` only affects citation expansion (mN -> member cells) and never enters the material
    text.
    R3' adjudication and R5 answering share this one renderer (so the material as seen by the two
    prompts is identical and cannot drift apart).
    Atoms do not go into the material: they are the retrieval unit (used by R1 for locating), and
    feeding them to the grader or the answerer would let them be read as a "reference answer" and
    muddy the judgment.
    """
    c = hit.cell
    return "\n".join([f"━━━ {handle} ━━━", cell_lead(c), c.episode or "(no episode)"])


# -- Material rendering order (the P1-B A/B switch) --
# relevance = the R2 rerank order (the current baseline); time_asc / time_desc = ordered by the cell
# time window.
# CONFLICT_RULE requires that "the later one overrides the earlier": in time order, the LLM's natural
# reading order IS the conflict-consumption order, so it no longer has to reorder the cells itself by
# reading the dates out of each header (one reordering mistake and it takes the stale value).
# The env var is read on every call (friendly to tests and to multi-arm comparisons); once the A/B is
# settled, change the default to the winner.
_R5_ORDER_ENV = "PERSONOS_R5_ORDER"
_R5_ORDER_FLOOR = datetime(1, 1, 1, tzinfo=timezone.utc)   # sort floor for a missing t_start (timezone-aware)


def _order_hits_for_answer(hits: list[CellHit], order: str) -> list[CellHit]:
    """Reorder the answering material by `order`; a stable sort — cells at the same time or with no
    time keep the rerank order (so relevance is still the reference for ties)."""
    if order == "relevance":
        return hits
    return sorted(hits, key=lambda h: ensure_aware(h.cell.t_start) or _R5_ORDER_FLOOR,
                  reverse=(order == "time_desc"))


def answer_from_cells(
    llm: ChatLLM, *, query: str, subject: str, hits: list[CellHit],
    now_dt: datetime | None = None,
    feedback: str = "", boundary: str = "", profile: str = "", scenario: str = "",
) -> MemoryAnswer:
    """R5 answering: the episode of each material unit (a plain cell or a woven memcell', which are
    structurally identical) is the only primary material, and the answer follows the rules.

    now_dt goes into the prompt as the CURRENT TIME anchor (the only basis for converting relative
    quantities in "how long ago" style questions).
    feedback: the fix instruction after adjudication returned a defect (the re-answer round);
    boundary: the boundary note for enumeration coverage (5b).
    With no material there is no LLM call (an empty answer means "nothing relevant"); a parse failure
    is passed through as-is; an infrastructure error (rate limit, timeout) returns an empty answer and
    lets the caller produce the no-answer account — the exception text is never used as the answer.
    """
    if not hits:
        logger.info(f"R5 answer: no materials, returning an empty answer directly q={query!r}")
        return MemoryAnswer(answer="")
    order = os.environ.get(_R5_ORDER_ENV, "").strip() or "relevance"
    hits = _order_hits_for_answer(hits, order)
    handles = [f"m{i + 1}" for i in range(len(hits))]   # the fast path's own mN handles for this window (the deep track's catalogue uses cN; the two are not shared)
    block = "\n\n".join(cell_block(h, hd) for h, hd in zip(hits, handles))
    bnd = f"\n\nBOUNDARY\n{boundary}" if boundary else ""
    fb = (f"\n\nJUDGE FEEDBACK\nA reviewer checked your previous draft against the SAME materials "
          f"and found: {feedback}\nRe-answer the question, fixing exactly what it points out."
          if feedback else "")
    subj = subject or "(not determined)"
    tnow = f"\n\nCurrent time: {(now_dt or now()).isoformat()}" if now_dt else ""
    prof = f"\n\n{profile}" if profile else ""       # only shapes how the answer is organized (detail, language); it never changes which facts are chosen
    user = (f"MEMORY MATERIALS\n{block}{bnd}{tnow}{fb}{prof}"
            f"\n\nQUESTION SUBJECT\n{subj}\n\nUSER QUESTION\n{query}")
    sys = with_scenario(_ANSWER_SYS, "# How to read the materials", scenario, _SCEN_DIR_ANSWER)
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    # mN -> the cell ids that unit covers (a woven unit expands to its full set of member cells, a
    # plain unit to itself); deduped across units while preserving order
    h2ids = {hd: (h.covers or [h.cell.id]) for hd, h in zip(handles, hits)}
    try:
        obj, raw = chat_json(llm, messages, max_tokens=2000, num_tries=3, stage="answer")
        answer = str(obj.get("answer") or "").strip()
        cited: list[str] = []
        for x in (obj.get("cells") or []):
            for cid in h2ids.get(str(x).strip(), []):
                if cid not in cited:
                    cited.append(cid)
        res = MemoryAnswer(answer=answer,
                           cited_cells=cited,   # unknown handles are dropped rather than crashing
                           system=sys, user=user, raw=raw)
    except Exception as e:
        # Only a JSON parse failure has .raw (the model did answer, the format was just bad) — pass it
        # through as-is; infrastructure errors like a rate limit or a timeout have no usable original
        # text, and the exception text must never become the answer (H1), so an empty answer goes up
        # to the caller's no-answer account
        raw = getattr(e, "raw", "")
        if raw:
            logger.warning(f"R5 answer parse failed, passing the raw text through as-is: {e}")
            res = MemoryAnswer(answer=raw, system=sys, user=user)
        else:
            logger.warning(f"R5 answer call failed (infrastructure/network), returning an empty answer for the caller to handle: {type(e).__name__}: {e}")
            res = MemoryAnswer(answer="", system=sys, user=user)
    logger.info(f"R5 answer cells={len(hits)} cited={len(res.cited_cells)} "
                f"ans_chars={len(res.answer)} q={query!r}")
    return res


# Chinese names for the epistemic status, shared by the answering material and the public view (the
# three object_type categories, which affect how the answer is worded).
_TYPE_CN = {"event": "经历", "fact": "事实", "claim": "说法"}
