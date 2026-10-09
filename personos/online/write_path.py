"""The write path (fused architecture §2): W0 persists each utterance as evidence -> W1 boundary
detection -> W2 cell generation.

The guiding principle of the write side: it only segments, weaves episodes and extracts atoms, and
performs no cross-cell reconciliation at all — duplicates are allowed to exist (a redundant index),
and conflicts are a matter for answering time (D1).
Batch writes during benchmarking and online product writes go through the same loop; the former just
fast-forwards through a whole conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

import numpy as np
from loguru import logger

from ..models import (
    ATOM_TEXT_SPEC,
    AXIS_MENU,
    DOMAIN_VOCAB,
    EvidenceRecord,
    EvidenceRef,
    MemCell,
    MemoryAtom,
    ensure_aware,
    normalize_domains,
    normalize_kind,
    now,
    vocab_menu,
)
from ..storage.atom_store import AtomStore
from ..storage.cell_store import CellStore
from ..storage.chain_store import ChainStore
from ..storage.evidence_store import EvidenceStore
from ..storage.seg_store import MemorySegStore, SegStore
from . import chain_build
from .events import EVENT_ADD, emit
from .llm import ChatLLM, chat_json, with_scenario

# Directives for caller-scenario injection (they only tune attention and level of detail; facts are
# never altered, invented, or omitted) -- see with_scenario
_SCEN_DIR_BOUNDARY = ("Use it as background for the caller's typical dialogue rhythm and what counts "
                      "as one coherent topic; it informs the judgment below but never overrides the "
                      "priority rules.")
_SCEN_DIR_EPISODE = ("Narrate the facts this caller cares about more fully and faithfully. This sets "
                     "emphasis (level of detail) only — still cover the whole segment, never drop a "
                     "stated fact.")
_SCEN_DIR_ATOM = ("Facts matching this focus are high-value retrieval anchors: make sure each DISTINCT "
                  "one is covered by an atom. This only raises priority within what is worth "
                  "remembering; it never licenses inventing facts absent from the transcript, dropping "
                  "other worth-remembering facts, nor logging restatements as separate atoms.")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


# Segment-length safety valve: the backstop for when W1 misjudges; past this many turns the segment is
# force-closed (fused architecture, W1 dimension 5)
MAX_SEGMENT_TURNS = 30
# The "most recent turns of the current segment" window in the W1 prompt: keeps a long segment from
# blowing up the boundary-detection prompt
_BOUNDARY_WINDOW = 6
# What happens when a W2 step 2 quote does not match: refs are left empty (the atom is still
# retrievable, and traceability goes through cell -> evidence)
_D_MENU = "- domains (D axis, 0-3 Dxx codes; never output the labels):\n  " + vocab_menu(DOMAIN_VOCAB)


# -- W0 - persist each utterance as evidence (no LLM, passive; evidence is no longer embedded — the
# whole store holds only two kinds of vector, atom and topic) --

def append_utterance(
    evidence_store: EvidenceStore, *, session_id: str, speaker: str, text: str,
    now_dt: Optional[datetime] = None,
    modality: str = "text", content_ref: Optional[str] = None, sha256: str = "",
    source_extra: Optional[dict] = None,
) -> str:
    rec = EvidenceRecord(holder=speaker, content_inline=text,
                         modality=modality, content_ref=content_ref, sha256=sha256,
                         source={"session_id": session_id, **(source_extra or {})},
                         captured_at=now_dt or now())
    evidence_id = evidence_store.append(rec)
    logger.debug(f"W0 evidence persisted ev={evidence_id} speaker={speaker} modality={modality} len={len(text)}")
    return evidence_id


def _transcript(records: list[EvidenceRecord]) -> str:
    """Render the raw utterances of a segment with speaker and timestamp (for the W1/W2 prompts, and
    so the LLM can write the dual-time format)."""
    lines = []
    for r in records:
        ts = r.captured_at.strftime("%Y-%m-%d %H:%M") if r.captured_at else "?"
        lines.append(f"[{ts}] {r.holder}: {r.content_inline or ''}")
    return "\n".join(lines)


# -- W1 - boundary detection (one small LLM call per utterance) --

@dataclass
class BoundaryDecision:
    should_end: bool
    confidence: float = 0.0
    topic_summary: str = ""    # a one-sentence topic for the current segment, used as a reference by W2 step 1's topic
    raw: str = ""              # for traceability


_BOUNDARY_SYSTEM = """# Role
You are the dialogue boundary detector in a memory write pipeline: decide whether a newly arrived
BATCH of utterances continues the current topic segment or opens a new one. The batch is atomic —
its utterances always stay together in the same segment (a question and its answer arriving in one
batch must never be split).

# Decision dimensions (by priority)
1. Substantive topic shift (highest priority): the new batch's core topic clearly differs from the
   current segment → end the segment.
2. Intent switch: the segment's core question is resolved and the new batch opens a brand-new
   task/intent → end.
3. Small talk / farewells are NOT shifts: closing phrases like "Thanks!", "Talk soon!" stay in the segment.
4. Time gap: hours or days since the previous utterance → end; only a few minutes → lean continue.
5. Segment length 3-20 turns: a long segment showing signs of drifting → lean end; under 3 turns,
   continue unless the topic clearly switched.

# Output (JSON only, no explanation)
{"should_end": true|false, "confidence": 0-to-1 decimal, "topic_summary": "one-sentence topic of the CURRENT segment"}
topic_summary states what the CURRENT segment is about, as a reference for downstream cell-topic
generation: one sentence making clear WHO is doing WHAT; keep names/brands/places/numbers/times
explicit and detailed — no pronouns, no generalization. Write it in the language of the dialogue."""


def detect_boundary(
    llm: ChatLLM, seg: list[EvidenceRecord], new_records: list[EvidenceRecord],
    *, gap_minutes: Optional[float] = None, scenario: str = "",
) -> BoundaryDecision:
    """Decide whether the newly arrived batch opens a new segment. With an empty seg there is no LLM
    call (the first batch of a new segment has no boundary to judge).

    new_records is one atomic batch as declared by the caller (say a QA pair): it is never split
    internally, and what is judged is the whole batch against the current segment. A parse failure or
    exception conservatively continues the segment (should_end=False): wrongly closing hurts episode
    coherence, whereas wrongly continuing only hurts segment length and the safety valve catches it.
    scenario: the caller's scenario description (may be empty) — injected to calibrate segmentation
    (the shape of the dialogue, and what counts as one topic).
    """
    if not seg:
        return BoundaryDecision(should_end=False)
    gap = "unknown" if gap_minutes is None else f"{gap_minutes:.0f} minutes"
    user = (
        f"Current segment turns: {len(seg)}\nTime gap since the previous utterance: {gap}\n\n"
        f"—— Current segment (most recent turns) ——\n{_transcript(seg[-_BOUNDARY_WINDOW:])}\n\n"
        f"—— Newly arrived batch ({len(new_records)} utterances) ——\n{_transcript(new_records)}"
    )
    sys = with_scenario(_BOUNDARY_SYSTEM, "# Decision dimensions (by priority)",
                        scenario, _SCEN_DIR_BOUNDARY)
    try:
        data, raw = chat_json(llm, [{"role": "system", "content": sys},
                                    {"role": "user", "content": user}], stage="boundary_detect",
                              max_tokens=200)
        d = BoundaryDecision(should_end=bool(data.get("should_end")),
                             confidence=float(data.get("confidence") or 0.0),
                             topic_summary=str(data.get("topic_summary") or ""), raw=raw)
        logger.info(f"W1 boundary end={d.should_end} conf={d.confidence:.2f} seg_len={len(seg)} "
                    f"gap={gap} summary={d.topic_summary!r}")
        return d
    except Exception as e:   # noqa: BLE001  a parse failure or network hiccup always continues conservatively (wrongly closing hurts the episode, while continuing is caught by the safety valve)
        logger.warning(f"W1 boundary detection failed, conservatively continuing the segment: {e}")
        return BoundaryDecision(should_end=False, raw=str(e))


# -- W2 - cell generation (two LLM calls per segment, in order) --

@dataclass
class CellBuild:
    """What generating one cell produced (including traceability)."""
    cell: MemCell
    atoms: list[MemoryAtom]
    gen: dict = field(default_factory=dict)   # {call1: {system,user,raw}, call2: {...}}, for workbench inspection
    chain_assign: "chain_build.ChainAssignResult | None" = None   # the W2.5 chain-assignment product (disabled or failed = None, or everything free-floating)


_EPISODE_SYSTEM = """# Role
You are the episode weaver in a memory write pipeline: weave one topic segment of dialogue into a
third-person narrative, plus a one-sentence topic and life-domain codes.

# Task (on the FULL segment transcript provided)
1. topic: one sentence stating what this segment is about — WHO is doing WHAT + key proper nouns kept
   verbatim (names/brands/places/numbers), no pronouns, no generalization (e.g. "Caroline preparing a
   joint exhibition with Rob"). The topic is this segment's retrieval face and material header — word
   it so a later search can hit it.
2. episode: a third-person narrative covering the whole segment in dialogue order — this is the primary
   material for all future answering.
3. domains: life-domain codes for this segment (0-3 D-axis codes).

# Hard constraints on episode (detail preservation carries most of the retrieval & answering score)
- Coverage first: before writing, walk the transcript utterance by utterance — every fact-bearing
  line (a date, time, amount, count, name, address, or one-off mention) must appear in the
  episode. Never trade a "small" fact (a departure day, a ticket price, a one-time plan) for
  narrative flow: write "departed on July 11", never "departed in mid-July".
- Full names, never pronouns: "Caroline and Rob", not "she and her friend".
- Proper nouns verbatim: brands, place names, book titles, restaurant names enter the text as-is.
- Numbers exact: "3 pizzas" not "some pizzas"; frequency specific: "every Tue & Thu" not "often".
- Dates/times of departures, appointments and deadlines, amounts, and identifiers enter the episode
  verbatim — a bare date is often the entire answer to a later question.
- Specific activities stay specific: "hot yoga" not "exercise".
- Dual time format: relative wording + absolute date written together — "last week (2026-08-18)".
- Complete when/who/where/how bounds: state clearly who did what and when; every qualifier present in
  the source (where / with whom / how / how often / until when) enters the narrative — never drop a
  stated qualifier, never invent an unstated dimension.
- Attribution explicit: make clear who said/did what; with multiple speakers, keep each person's
  matters distinct.
- Language follows the source dialogue: write topic and episode in the SAME language as the dialogue.
- Only what was said in this segment: no inference, no addition; every narrative sentence must trace
  back to the transcript.

""" + _D_MENU + """

# Output (JSON only)
{"topic": "...", "episode": "...", "domains": ["D13"]}"""


_ATOM_SYSTEM = """# Role
You are the atomic-memory extractor in a memory write pipeline: batch-extract atomic facts (atoms)
from one dialogue segment.

# Input
An episode (narrative — navigate by it) plus the FULL segment transcript (the ground truth — decide by
it): speaker attribution, times and numbers are always settled from the transcript.

# What to extract
- Worth remembering: stable identity & attributes / long-term preferences & habits / relationships /
  goals, plans, commitments & schedules (clear intent or a time attached — including near-term
  bookings, useful until they happen) / constraints & rules / important events & experiences / stable
  facts about the people and things that matter to someone / concrete items of interest (books read,
  places visited, activities enjoyed, family members' likes — record even passing mentions).
- Not worth remembering: passing whims & momentary desires / small talk / immediate instructions about
  the current task / transition chatter.
- Read question-and-answer exchanges as a whole: "guess where I live" + "Shanghai?" + "yes" → "lives
  in Shanghai" holds. The assistant's words are context only — its suggestions/guesses count as fact
  only when the user explicitly endorses them.
- Multiple speakers: attribute every statement to its owner — holder is the speaker the proposition
  TRULY belongs to (the name used in the dialogue; the person talking to the assistant is "user").
  Never attribute someone else's experience to user.
- A question or follow-up is not itself a fact; an unanswered cliffhanger is not extracted.

# Coverage as retrieval anchors (atoms ARE the retrieval face — the episode already carries the full
# narrative for answering; atoms exist so a later query can FIND this segment, NOT to re-log every line)
- One atom per DISTINCT fact: give each distinct worth-remembering thing exactly ONE atom. Together the
  atoms are MECE — no two restate the same fact, and collectively they cover every distinct
  worth-remembering point in the segment. Do NOT chase completeness by adding near-duplicate atoms.
- Merge repetition; never one atom per utterance: the same fact stated, rephrased, reacted to or argued
  over across many lines is still ONE atom. Blow-by-blow narration — every gesture, each repeated
  complaint, the back-and-forth bickering, momentary reactions — belongs in the EPISODE, not in atoms.
- **Always return at least one atom — an empty list is never a valid answer.** How many is decided by
  the segment itself: one atom per DISTINCT worth-remembering fact. When the segment genuinely carries
  no durable fact (pure greetings, acknowledgements), return exactly ONE atom that states in one
  sentence what happened in this segment. An argument is still a source of facts — what got damaged,
  who owns what, stated numbers, majors, habits — extract those even when the surrounding lines are
  repetitive bickering. Reason: atoms are the ONLY retrieval entry to this segment; return none and
  the whole segment becomes permanently unreachable, however rich its episode is.
- Skip process trivia: small talk, filler, transient reactions, and repeated micro-actions get NO atom;
  they are neither searched for nor worth remembering.
- Distinct items & attributes DO each get their own atom — they are separate search anchors, NOT
  repetition. A list of things (bought / visited / people present) → one atom each. Parallel attributes
  of one subject that are each asked about on their own — schedule, venue, coach, price, a stated
  frequency/count ("every Tue & Thu", "the third time") — → separate atoms; a bare name / address /
  number is often exactly what gets looked up later. This is coverage of distinct facts, the opposite
  of logging restatements. (A qualifier bounding one occurrence — its when/where — stays inside its atom.)

# Fields per atom
- text: follow the spec below (third-person single sentence, self-contained subject-verb-object, dual
  time format, proper nouns & numbers verbatim).
- object_type (epistemic state; shapes future answer wording): fact = normalized persistent state
  ("Caroline lives in Shanghai"); claim = opinion/intention/plan/hearsay (always attributed wording,
  e.g. "Caroline said she plans to hold the exhibition next month"); event = a one-time occurrence
  with a definite moment ("Caroline sprained her right ankle last week (2026-08-18)").
- holder: who said it / whose attribute (the name from the dialogue, or "user").
- kind: one K-axis code.
- domains: 0-3 D-axis codes.
- when: the date the fact itself happened or holds (ISO yyyy-MM-dd, e.g. "2026-08-18") — the day it
  HAPPENED, not the day it was talked about. Resolve relative words (yesterday / last Wednesday /
  next month) into absolute dates against the transcript timestamps. Anchor to the time the fact
  refers to, NOT the dialogue day: "made a limited edition line last week" said on 2023-08-23 → when
  is a date within 08-16 ~ 08-22 (the week BEFORE the dialogue), never 08-23. For persistent states
  with no intrinsic date, give null (code falls back to the dialogue date, meaning "already true as
  of that day").
- quote: the atom's single most relevant source sentence, a CONTIGUOUS SUBSTRING copied verbatim from
  the transcript (used only to link back to the evidence — an anchor, NOT required to cover the whole
  atom text; when the atom merges several lines, pick the one line that best evidences it). Never
  rewrite it; if no single line fits, leave it empty.

""" + ATOM_TEXT_SPEC + "\n\n" + AXIS_MENU + """

# Output (JSON only; output {"atoms":[]} when nothing is worth remembering)
{"atoms":[{"text":"...","object_type":"fact","holder":"user","kind":"K01","domains":["D01"],"when":"2026-08-18","quote":"..."}]}"""


def _match_evidence_refs(records: list[EvidenceRecord], quote: str) -> list[EvidenceRef]:
    """A quote substring -> the evidence it hits (verbatim containment counts; if several utterances
    match, take them all in time order).

    No match -> leave refs empty and warn: the atom is still retrievable, but the gold evidence chain
    is broken at the link-back step, and when benchmarking counts breakpoints that has to be countable
    from the logs.
    """
    q = (quote or "").strip()
    if not q:
        return []
    refs = [EvidenceRef(evidence_id=r.id)
            for r in records if q in (r.content_inline or "")]
    if not refs:
        logger.warning(f"W2(2) quote did not match any original utterance, refs left empty (evidence chain break) quote={q[:40]!r}")
    return refs[:3]   # one atom's source is at most a few utterances; this stops the LLM handing back the whole transcript


# Episode classification: only appended when the caller passed a task_type vocabulary (otherwise the
# LLM never learns the field exists and it defaults to unknown).
# It rides on the W2 step 1 call, so it costs no extra LLM call. A value outside the list falls back
# to unknown during parsing.
_CLASSIFY_SUFFIX = """

# Episode classification (REQUIRED extra field)
Also classify this whole segment into EXACTLY ONE type from this list: {types}
Add an "episode_type" field to your JSON with the chosen value verbatim; if none fits, use "unknown".
Output JSON: {{"topic": "...", "episode": "...", "domains": [...], "episode_type": "..."}}"""


def build_cell(
    llm: ChatLLM, embedder: Embedder,
    evidence_store: EvidenceStore, cell_store: CellStore, atom_store: AtomStore,
    records: list[EvidenceRecord], *, session_id: str, topic_hint: str = "",
    chain_store: ChainStore | None = None, task_type: list[str] | None = None,
    scenario: str = "",
) -> CellBuild:
    """W2: one closed segment of raw utterances -> a MemCell (topic / episode / domains) + atoms,
    embedded in one batch and persisted.

    Call 1 failing degrades to using the raw transcript as the episode (no information is lost, only
    the formatting); call 2 failing persists the cell with no atoms (a missed extraction heals itself
    later through the deep track's remember or a subsequent dream pass, see D3/D8).
    After the atoms are persisted, W2.5 chain assignment runs (deriving chain_store from atom_store
    when it was not injected); a failed assignment is non-blocking — this cell's atoms are left
    free-floating and the already-persisted cell and atoms are never touched.
    """
    transcript = _transcript(records)
    gen: dict = {}

    # Call 1: topic + episode + cell domains (a missing timestamp renders as "?", so formatting cannot
    # blow the whole of W2 apart)
    t0s = (records[0].captured_at.strftime("%Y-%m-%d %H:%M") if records[0].captured_at else "?")
    t1s = (records[-1].captured_at.strftime("%Y-%m-%d %H:%M") if records[-1].captured_at else "?")
    user1 = (f"Session window: {t0s} ~ {t1s}\n"
             + (f"(session topic: {topic_hint})\n" if topic_hint else "")
             + f"\n—— Full segment transcript ——\n{transcript}")
    types = [t for t in (task_type or []) if isinstance(t, str) and t.strip()]
    sys1 = with_scenario(_EPISODE_SYSTEM, "# Task (on the FULL segment transcript provided)",
                         scenario, _SCEN_DIR_EPISODE)
    sys1 += (_CLASSIFY_SUFFIX.format(types=types) if types else "")
    topic, episode, cell_domains, episode_type = "", "", [], "unknown"
    try:
        data, raw = chat_json(llm, [{"role": "system", "content": sys1},
                                    {"role": "user", "content": user1}], max_tokens=3000,
                              stage="episode_weave")
        topic = str(data.get("topic") or "").strip()
        episode = str(data.get("episode") or "").strip()
        cell_domains = normalize_domains(data.get("domains"))[:3]
        if types:   # only honoured when a vocabulary was passed: the LLM's output must be in the list, otherwise it falls back to unknown (nothing is force-fitted)
            et = str(data.get("episode_type") or "").strip()
            episode_type = et if et in set(types) else "unknown"
        gen["call1"] = {"system": sys1, "user": user1, "raw": raw}
    except Exception as e:   # noqa: BLE001  a parse failure or network hiccup degrades to the raw transcript as the episode (no information lost, only the formatting)
        logger.warning(f"W2(1) failed, degrading the episode to the raw transcript: {e}")
        topic = topic_hint or (records[0].content_inline or "")[:50]
        episode = transcript
        gen["call1"] = {"system": sys1, "user": user1, "raw": str(e)}

    cell = MemCell(session_id=session_id, topic=topic, episode=episode, domains=cell_domains,
                   episode_type=episode_type,
                   t_start=records[0].captured_at, t_end=records[-1].captured_at,
                   evidence_refs=[EvidenceRef(evidence_id=r.id) for r in records])

    # Call 2: batch atom extraction (fed both the episode and the raw transcript)
    atoms: list[MemoryAtom] = []
    user2 = (f"—— Episode (episodic memory compressed from the original dialogue) ——\n{episode}\n\n"
             f"—— Full original transcript ——\n{transcript}")
    sys2 = with_scenario(_ATOM_SYSTEM, "# Input", scenario, _SCEN_DIR_ATOM)
    seg_date = records[0].captured_at

    def _extract(nudge: str = "") -> tuple[list[MemoryAtom], str]:
        msgs = [{"role": "system", "content": sys2}, {"role": "user", "content": user2}]
        if nudge:   # append a correction when re-extracting: resending the identical request just rolls the same empty-biased dice again
            msgs += [{"role": "assistant", "content": '{"atoms":[]}'},
                     {"role": "user", "content": nudge}]
        data, raw_out = chat_json(llm, msgs, max_tokens=4000, num_tries=3, stage="atom_extract")
        out: list[MemoryAtom] = []
        for item in (data.get("atoms") or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            out.append(MemoryAtom(
                memcell_id=cell.id,
                object_type=item.get("object_type") if item.get("object_type") in ("fact", "claim", "event") else "claim",
                text=text,
                holder=str(item.get("holder") or "user"),
                kind=normalize_kind(item.get("kind")) or "K01",
                domains=normalize_domains(item.get("domains"))[:3],
                occurrence_time=ensure_aware(item.get("when")) or seg_date,
                evidence_refs=_match_evidence_refs(records, str(item.get("quote") or "")),
            ))
        return out, raw_out

    try:
        atoms, raw = _extract()
        # Zero atoms means extracting once more: atoms are the **retrieval antennae**, and having none
        # makes this cell unreachable by any query — an island (the episode is still there, but
        # nothing can find it). chat_json's num_tries only covers JSON parse failures, so a
        # well-formed {"atoms":[]} sails straight through. In practice the same input has returned 0
        # in one process and 2 in another, which says an empty result is quite likely a missed
        # extraction rather than "there really is nothing to remember". If the second attempt is also
        # empty, we accept that there really is nothing to remember.
        if not atoms:
            logger.warning(f"W2(2) extracted 0 atoms, re-extracting with a correction note turns={len(records)} topic={topic[:30]!r}")
            atoms, raw = _extract(
                "You returned an empty list last time. Re-read the segment: even when it is "
                "mostly argument or small talk, any **persistent facts** it mentions "
                "(someone's belongings, explicit numbers, identity/profession/habits, "
                "concrete events that happened) must still be extracted. Only return an "
                "empty list when the segment is truly nothing but greetings and replies "
                "with no fact worth remembering. Output the JSON body only.")
        gen["call2"] = {"system": sys2, "user": user2, "raw": raw}
    except Exception as e:   # noqa: BLE001  a parse failure or network hiccup -> persist the cell with no atoms (a later re-extraction heals it, see D3/D8)
        logger.warning(f"W2(2) failed, atoms left empty: {e}")
        gen["call2"] = {"system": sys2, "user": user2, "raw": str(e)}

    # -- Backstop: a memcell must have at least one retrieval antenna --
    # The fast path's R1 (retrieval.search_atoms) only searches **atom** vectors; a cell's
    # topic_embedding is used only by the deep track, over an already-selected subset of cells. So a
    # cell with zero atoms is **completely unreachable** on the fast path — its episode is still in
    # the store, but no phrasing of a query can retrieve it (four phrasings were tried, all missed).
    # However thin a segment is, it deserves one summary sentence as an antenna rather than vanishing
    # entirely. The topic is already a one-sentence summary of "who is doing what", so it is used
    # directly as this atom's body; evidence_refs points at the whole segment, keeping traceability
    # intact.
    # Marking source="w2_fallback" makes it easy to count afterwards how often this path fires (the
    # normal path is always "w2").
    if not atoms:
        summary = (topic or "").strip() or (episode or "").strip()[:120]
        if summary:
            atoms = [MemoryAtom(
                memcell_id=cell.id, object_type="event", text=summary,
                holder="user",          # a segment summary belongs to no single person, so use the default holder
                kind="K04",             # event/experience: what happened during this segment
                domains=cell_domains, occurrence_time=seg_date, source="w2_fallback",
                evidence_refs=[EvidenceRef(evidence_id=r.id) for r in records])]
            logger.warning(f"W2(2) still 0 atoms after re-extraction -> synthesizing 1 summary atom as a "
                           f"backstop (otherwise this cell is permanently unrecallable) "
                           f"turns={len(records)} text={summary[:60]!r}")
        else:
            logger.error(f"W2(2) 0 atoms and both topic and episode are empty, this cell will be unrecallable cell={cell.id}")

    # Batch embed: one topic + one per atom (the only two kinds of vector in the whole store) in a
    # single call; an empty topic takes no slot
    texts = ([topic] if topic else []) + [a.text for a in atoms]
    vecs = list(embedder.embed(texts)) if texts else []
    topic_vec = np.asarray(vecs[0]) if topic and vecs else None
    atom_vecs = vecs[1:] if topic else vecs

    cell_store.upsert(cell, topic_embedding=topic_vec)
    if atoms:
        atom_store.upsert_many(list(zip(atoms, atom_vecs)))
    logger.info(f"W2 cell persisted id={cell.id} turns={len(records)} atoms={len(atoms)} "
                f"episode_type={episode_type} domains={cell_domains}\n"
                f"  topic={topic!r}\n"
                f"  episode={episode!r}\n"
                f"  atoms(n={len(atoms)})=" + "\n".join(
                    f"    [{i}] {a.object_type}/{a.kind} {a.text!r} holder={a.holder} "
                    f"domains={a.domains} occ={a.occurrence_time}" for i, a in enumerate(atoms, 1)))
    # Broadcast "these memories were just formed". The memory service does not care who is listening —
    # delivery, signing and subscription relationships are all the application layer's business. With
    # no subscribers it costs nothing, and a failing callback does not affect this write.
    if atoms:
        emit(
            atom_store.user_id,
            EVENT_ADD,
            {
                "cell_id": cell.id,
                "session_id": cell.session_id,
                "atom_ids": [a.id for a in atoms],
                "count": len(atoms),
                # Ids only, no body text: the body belongs to the memory itself, and a subscriber that wants the content calls back into the API for it
                "domains": sorted({d for a in atoms for d in (a.domains or [])}),
            },
        )

    # W2.5 chain assignment (only runs when there are atoms; failures are absorbed inside
    # assign_chains and never block)
    chain_assign = None
    if atoms:
        cs = chain_store or ChainStore(atom_store.db, atom_store.user_id)
        try:
            chain_assign = chain_build.assign_chains(
                llm, cs, list(zip(atoms, atom_vecs)), origin_cell_id=cell.id)
            gen["w25"] = {"system": chain_build._ASSIGN_SYSTEM, "result": {
                "new": [c.id for c in chain_assign.new_chains],
                "appended": chain_assign.appended, "free": len(chain_assign.free)}}
        except Exception as e:   # noqa: BLE001  belt and braces: assign_chains already catches its own errors, and this covers any unknown path
            logger.warning(f"W2.5 chain assignment raised, this cell's atoms are left unchained: {e}")
    return CellBuild(cell=cell, atoms=atoms, gen=gen, chain_assign=chain_assign)


# -- Orchestration: SessionWriter (online product writes go utterance by utterance, benchmark batches
# fast-forward, both through the same loop) --

@dataclass
class StepResult:
    """All the intermediate state after feeding one utterance (for scenario and workbench inspection)."""
    evidence_id: str
    record: EvidenceRecord
    boundary: Optional[BoundaryDecision] = None   # None = the first utterance of a segment (no boundary to judge) or a forced close at 30 turns
    forced_close: bool = False                    # True = closed by the safety valve, not by the LLM's judgment
    closed_cell: Optional[CellBuild] = None       # the cell this utterance closed (the new utterance belongs to the new segment)


@dataclass
class FeedMsg:
    """One atomic member of a feed_batch (one message: speaker + text and/or image)."""
    speaker: str
    text: str = ""
    image: Optional[bytes] = None
    image_content_type: str = "image/jpeg"


@dataclass
class BatchStepResult:
    """The intermediate state after one feed_batch: the batch is atomic — either the whole batch joins
    the old segment, or the whole batch opens a new one."""
    evidence_ids: list[str]
    records: list[EvidenceRecord]
    boundary: Optional[BoundaryDecision] = None   # None = the first batch of a segment (no boundary to judge) or a forced close by the safety valve
    forced_close: bool = False                    # True = closed by the safety valve, not by the LLM's judgment
    closed_cell: Optional[CellBuild] = None       # the cell this batch closed (what closes is the old segment, which does not include this batch)
    # Things that partially succeeded. A write is never rejected for a missing
    # optional capability, but it must not silently do less than asked either —
    # an image stored without understanding contributes nothing to retrieval,
    # and the caller deserves to know that rather than discover it at recall.
    warnings: list[str] = field(default_factory=list)


class SessionWriter:
    """The write state machine for one session: fed utterance by utterance, building a cell through W2
    whenever a segment closes, and force-closing at the end of the session.

    Made stateless: the open segment lives in seg_store (the service injects a Redis implementation,
    so it is shared across replicas and survives a redeploy; the default is a process-private instance,
    which keeps single-process script semantics unchanged). This instance only accumulates `cells` —
    a progress view for batch scripts; the service creates a new writer per request, so that list
    lives and dies with the request rather than sticking around.
    """

    def __init__(self, llm: ChatLLM, embedder: Embedder,
                 evidence_store: EvidenceStore, cell_store: CellStore, atom_store: AtomStore,
                 *, session_id: str, user_id: str = "", max_turns: int = MAX_SEGMENT_TURNS,
                 seg_store: SegStore | None = None, chain_store: ChainStore | None = None,
                 media_store=None, mllm=None):
        self.llm = llm
        self.embedder = embedder
        self.evidence_store = evidence_store
        self.cell_store = cell_store
        self.atom_store = atom_store
        self.chain_store = chain_store   # None means build_cell derives it from atom_store (same DB, same user)
        self.session_id = session_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.seg_store = seg_store or MemorySegStore()   # not injected = private in-process segment state (single-process semantics)
        self.cells: list[CellBuild] = []                 # the cells this instance produced (a progress view for batch scripts)
        # Image input dependencies (optional; without them images are unsupported and the pure-text
        # path is unaffected)
        self.media_store = media_store
        self.mllm = mllm

    @property
    def seg(self) -> list[EvidenceRecord]:
        """The current open segment (read live from seg_store as a copy; kept for compatibility with
        older callers' assertions)."""
        return self.seg_store.load(self.user_id, self.session_id)

    def _ingest_image(self, image: bytes, content_type: str, speaker: str,
                      context: Optional[list[EvidenceRecord]] = None) -> tuple[Optional[str], str, str]:
        """Store the image and look at it with context. Returns (content_ref, sha256, the image
        understanding text); a failure at any step degrades gracefully.

        The viewing purpose is the dialogue already in the current segment plus the utterances of this
        batch prepared so far — so the MLLM looks at the image knowing what this segment is about,
        rather than describing it aimlessly. If storage fails, we still try to look at the image from
        the in-memory bytes (the original is not kept, but the understanding is not lost).
        """
        content_ref, sha256 = None, ""
        if self.media_store is not None:
            try:
                stored = self.media_store.save_image(
                    image, owner=self.user_id or "user", content_type=content_type)
                content_ref, sha256 = stored.key, stored.sha256
            except Exception as e:   # noqa: BLE001  a storage failure does not block: we can still look at the image, we just keep no copy
                logger.warning(f"failed to store the image in the object store, no copy kept (understanding is still attempted): {e}")
        img_text = ""
        if self.mllm is not None and getattr(self.mllm, "available", False):
            ctx = _transcript((context or [])[-_BOUNDARY_WINDOW:]) if context else ""
            # English, so that an image description comes back in the library's
            # default language rather than following the prompt's. The model
            # still describes content in whatever language the image contains.
            purpose = (f"This conversation is about:\n{ctx}\n\nWith that context, "
                       "state the facts in the image that relate to it."
                       if ctx else "State the facts visible in the image: people, text, "
                                   "scene, countable objects.")
            img_text = self.mllm.look_image(image, purpose, content_type=content_type)
        return content_ref, sha256, img_text

    def _prepare(self, m: FeedMsg, *, now_dt: datetime,
                 source_extra: dict | None, context: list[EvidenceRecord]) -> EvidenceRecord:
        """W0: one message -> evidence persisted + an in-memory record. A message with an image first
        goes through storing and viewing it (failures degrade and never block)."""
        modality, content_ref, sha256 = "text", None, ""
        content_inline = m.text
        if m.image:
            # An image means the modality is image/mixed, regardless of whether storage succeeded:
            # even with no stored original, this is still an image message
            modality = "mixed" if m.text.strip() else "image"
            content_ref, sha256, img_text = self._ingest_image(
                m.image, m.image_content_type, m.speaker, context)
            if img_text:
                # The image understanding text is merged into the utterance: the user's caption first,
                # the observed facts after (so W1 and W2 read them together)
                content_inline = (m.text + "\n" if m.text.strip() else "") + f"[image] {img_text}"
        # The merged source goes into the persisted record too, not just the in-memory one —
        # otherwise a replay from the store would lose source_extra (e.g. video clip metadata)
        evidence_id = append_utterance(self.evidence_store, session_id=self.session_id,
                                       speaker=m.speaker, text=content_inline, now_dt=now_dt,
                                       modality=modality, content_ref=content_ref, sha256=sha256,
                                       source_extra=source_extra)
        return EvidenceRecord(id=evidence_id, holder=m.speaker, content_inline=content_inline,
                              modality=modality, content_ref=content_ref, sha256=sha256,
                              source={"session_id": self.session_id, **(source_extra or {})},
                              captured_at=now_dt)

    def feed_batch(self, msgs: list[FeedMsg], *,
                   now_dt: Optional[datetime] = None,
                   source_extra: dict | None = None,
                   task_type: list[str] | None = None,
                   scenario: str = "") -> BatchStepResult:
        """Feed one whole batch (atomically): W0 persists the whole batch as evidence -> (when the
        segment is non-empty) W1 judges the batch as a whole -> W2 runs if it closes -> the batch
        joins the segment.

        A batch now plays the role one utterance used to: either the whole batch joins the current
        segment, or the whole batch opens a new one — the inside of a batch (say a QA pair) is never
        split. Messages with images are stored and viewed one by one, and the viewing context
        includes the utterances of this batch prepared so far.
        """
        now_dt = now_dt or now()
        recs: list[EvidenceRecord] = []
        # During W0 the segment cannot change (the write lock is held), so one read is enough; the
        # image-viewing context is the current segment plus the utterances of this batch prepared so far
        ctx_seg = self.seg_store.load(self.user_id, self.session_id)
        for m in msgs:
            recs.append(self._prepare(m, now_dt=now_dt, source_extra=source_extra,
                                      context=ctx_seg + recs))

        seg = ctx_seg   # W0 only writes to the evidence store and does not touch the segment, so reuse it
        boundary, forced, closed = None, False, None
        if seg:
            # Safety valve: the segment would be over the limit after adding this batch
            # -> close the old segment first and then take the batch, with no LLM call
            if len(seg) + len(recs) > self.max_turns:
                forced = True
            else:
                gap = ((now_dt - seg[-1].captured_at).total_seconds() / 60
                       if now_dt and seg[-1].captured_at else None)
                boundary = detect_boundary(self.llm, seg, recs, gap_minutes=gap, scenario=scenario)
            if forced or boundary.should_end:
                closed = build_cell(self.llm, self.embedder, self.evidence_store,
                                    self.cell_store, self.atom_store, seg,
                                    session_id=self.session_id,
                                    topic_hint=boundary.topic_summary if boundary else "",
                                    chain_store=self.chain_store, task_type=task_type,
                                    scenario=scenario)
                self.cells.append(closed)
                seg = []                             # closing clears the segment; the save below overwrites the key wholesale
        self.seg_store.save(self.user_id, self.session_id, seg + recs)
        return BatchStepResult(evidence_ids=[r.id for r in recs], records=recs,
                               boundary=boundary, forced_close=forced, closed_cell=closed)

    def feed(self, speaker: str, text: str, *,
             now_dt: Optional[datetime] = None, source_extra: dict | None = None,
             image: bytes | None = None, image_content_type: str = "image/jpeg",
             task_type: list[str] | None = None, scenario: str = "") -> StepResult:
        """Feed one utterance (a single-message convenience wrapper, equivalent to a feed_batch of one;
        kept for internal scripts and benchmarks)."""
        r = self.feed_batch([FeedMsg(speaker=speaker, text=text, image=image,
                                     image_content_type=image_content_type)],
                            now_dt=now_dt, source_extra=source_extra, task_type=task_type,
                            scenario=scenario)
        return StepResult(evidence_id=r.evidence_ids[0], record=r.records[0],
                          boundary=r.boundary, forced_close=r.forced_close,
                          closed_cell=r.closed_cell)

    def end_session(self, *, task_type: list[str] | None = None,
                    scenario: str = "") -> list[CellBuild]:
        """End of session: force-close the open segment (this is what guarantees W1's segmentation
        covers the entire session). task_type is used to classify the trailing segment's episode."""
        seg = self.seg_store.load(self.user_id, self.session_id)
        if seg:
            closed = build_cell(self.llm, self.embedder, self.evidence_store,
                                self.cell_store, self.atom_store, seg,
                                session_id=self.session_id, chain_store=self.chain_store,
                                task_type=task_type, scenario=scenario)
            self.cells.append(closed)
            self.seg_store.clear(self.user_id, self.session_id)
            logger.info(f"session end forced close session={self.session_id} cells={len(self.cells)}")
        return self.cells
