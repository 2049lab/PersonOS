"""Unified construction of session dialogue history: rolling compaction + the last few turns verbatim.

This replaces the "take the last N turns" plus ad-hoc compaction that ingest, recall and arbitrate
each used to do on their own, unifying it in one place:
  - each session keeps a single ROLLING SUMMARY (overwritten in place, persisted);
  - each build = the summary + the entire tail since the last compaction;
  - when len(summary) + len(tail minus the last keep_recent turns) exceeds cap -> compact
    "old summary + aged turns" into a new summary with a target length of about cap/10; if it is
    still over, recompact (at most k times), and failing that hard-truncate by length;
  - the last keep_recent turns are always kept verbatim and never counted against cap (which stops
    one giant message from making compaction impossible).

The summary serves dialogue continuity, reference resolution and the user's current request — it is
not long-term fact extraction, which is the job of reconcile -> atom. The summary is a derived cache;
the evidence is kept as-is.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from ..models import EvidenceRecord
from ..storage.evidence_store import EvidenceStore
from ..storage.session_store import SessionContextStore

DEFAULT_CAP = 100000       # character cap on the history context (excluding the recent turns kept verbatim)
DEFAULT_KEEP_RECENT = 3    # how many recent turns are always kept verbatim during compaction
_MAX_RECOMPRESS = 2        # how many times to recompact when the result is still over cap, before hard truncation
_SUMMARY_HOLDER = "summary"
_VIDEO_HOLDER = "video"    # the holder used for video segments in the history (one recording = one episode turn)


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> str: ...


_COMPACT_SYS = """# Role
You are compressing the conversation history between a personal assistant and its user, to preserve context continuity for future conversations.
Note: you are NOT extracting long-term facts (that is the memory store's job) — you keep only the context needed to continue the conversation.

# Task
Compress the given conversation history into one coherent summary, targeting roughly {target} characters (stay under it if possible).
Cover the history's features across these 9 dimensions, as MECE as you can (write what is there, skip what is not — never pad):
1. Who the user is / basic identity and current situation (only what appears in the dialogue)
2. The user's current main needs, goals and intents (including implicit ones)
3. The user's preferences, habits, constraints and taboos
4. Key people / places / things / times — entities (for later reference resolution)
5. Fact changes and corrections (moving, renaming, preference changes, …): write them as "X→Y" with the absolute date of the change
6. The user's emotions, attitudes and feedback (especially corrections, dissatisfaction, approval directed at the assistant)
7. Topics already discussed / resolved, and their conclusions
8. Open, pending topics or tasks to follow up
9. The topic being discussed right now (to carry the conversation forward)

# Requirements
- Keep specific names and numbers.
- Times always as absolute dates — convert "last week / a few days ago / last month" into absolute dates (e.g. "2026-07") using the [date] prefixed to each line; no relative time expressions may survive in the summary (the summary itself carries no time anchor — a relative phrase can never be restored later).
- Write in the third person ("the user ...").
- Language follows the source dialogue.
- Output only the summary body: no title, no numbered points, no explanations."""


def _pair_turns(recs: list[EvidenceRecord]) -> list[list[tuple[str, str, str]]]:
    """Pair time-ordered evidence into "turns": a non-assistant utterance (user / third_party) opens a
    turn, and the assistant utterance that immediately follows joins that same turn.
    Each line carries a date (the date of captured_at) — compaction uses it to anchor phrases like
    "last week" or "a few days ago" to absolute dates."""
    turns: list[list[tuple[str, str, str]]] = []
    for r in recs:
        d = r.captured_at.date().isoformat() if r.captured_at else "?"
        line = (r.holder, r.content_inline or "", d)
        if r.holder == "assistant" and turns:
            turns[-1].append(line)
        else:
            turns.append([line])
    return turns


def _video_turns(video_recs: list[EvidenceRecord], session_id: str,
                 cell_store: Any) -> list[list[tuple[str, str, str]]]:
    """Video segments -> one `(video, episode)` turn per video memcell.

    Why not line by line: what a video flush writes into evidence is the **script, line by line** (a
    single 60s clip already produces 20+ lines), where the holder is a person's name, captured_at is
    unset, and content_inline of the raw_clip entries is still empty — dropping that straight into
    the history gives you 20+ verbatim dialogue turns plus a few empty ones: noisy and with no
    narrative. The episode is what this recording "was about".

    How we tell which cells are video cells: by **whether their evidence_refs intersect the video
    evidence ids**, not just by session — the same session also produces memcells for text segments,
    whose lines are already in the history verbatim, so including them again would duplicate them.

    Ordering: video evidence is only written in one batch by the flush at session_end, always after
    the text turns, so these turns are appended at the end of the text turns (matching the order
    by_session returns) and never get inserted in the middle of the text to disturb the `covered`
    watermark.
    """
    if cell_store is None or not video_recs:
        return []
    vids = {r.id for r in video_recs if r.id}
    turns: list[list[tuple[str, str, str]]] = []
    try:
        cells = cell_store.list_session(session_id)
    except Exception as e:  # noqa: BLE001  failing to read cells must not block history construction (the degraded case is that this segment has no context)
        logger.warning(f"failed to read session memcells, video segments will not enter the history: {e}")
        return []
    for cell in cells:
        refs = {getattr(r, "evidence_id", "") for r in (cell.evidence_refs or [])}
        if not (refs & vids):
            continue                     # a text-segment cell: its lines are already in the history verbatim
        ep = (cell.episode or "").strip()
        if not ep:
            continue
        d = cell.t_start.date().isoformat() if cell.t_start else "?"
        turns.append([(_VIDEO_HOLDER, ep, d)])
    return turns


def _turn_len(turn: list[tuple[str, str, str]]) -> int:
    return sum(len(t) for _, t, _ in turn)


def _out_lines(turns: list[list[tuple[str, str, str]]]) -> list[tuple[str, str]]:
    """History output for downstream: just (holder, text) (downstream resolves relative time against
    now_dt, so verbatim turns need no inline date)."""
    return [(h, t) for turn in turns for (h, t, _) in turn]


def _dated_block(turns: list[list[tuple[str, str, str]]]) -> str:
    """Render the compaction input as `[date] holder: text` — this gives the LLM a time anchor so it
    can convert relative times into absolute ones."""
    return "\n".join(f"[{d}] {h}: {t}" for turn in turns for (h, t, d) in turn)


def _aged_text(turns: list[list[tuple[str, str, str]]]) -> str:
    """Plain text, used for the concatenation fallback when compaction fails."""
    return " ".join(t for turn in turns for (_, t, _) in turn)


def _compact(llm: ChatLLM, prior_summary: str, aged: list[list[tuple[str, str]]], target: int) -> str:
    """Compact [old summary + aged turns] into one new summary. On failure it degrades to
    concatenating the old summary and the raw text (the caller truncates as a backstop)."""
    block = ""
    if prior_summary:
        block += f"Existing summary (earlier history):\n{prior_summary}\n\n"
    block += ("Earlier dialogue (the [date] before each line is that utterance's absolute "
              "date):\n" + _dated_block(aged))
    sys = _COMPACT_SYS.replace("{target}", str(target))
    try:
        out = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": block}],
                       temperature=0.2).strip()
        return out or block
    except Exception as e:  # noqa: BLE001  a compaction failure must not block the main path
        logger.warning(f"session history compaction failed, falling back to plain concatenation: {e}")
        return (prior_summary + " " + _aged_text(aged)).strip()


def build_history(
    evidence_store: EvidenceStore,
    session_id: str,
    *,
    llm: ChatLLM,
    cap: int = DEFAULT_CAP,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    target: int | None = None,
    cell_store: Any = None,
) -> list[tuple[str, str]]:
    """Build the dialogue history fed downstream = [summary?] + the tail verbatim. Rolls the
    compaction and persists it when needed.

    Note: ingest must call this BEFORE appending the current message, so the in-flight message is not
    counted into the history.

    cell_store: when given, **video segments are folded into their episode** (see _video_turns);
    when omitted, video lines go through verbatim (preserving the old behaviour, backward compatible).
    """
    store = SessionContextStore(evidence_store.db, user_id=getattr(evidence_store, "user_id", ""))
    summary, covered = store.get(session_id)
    recs = evidence_store.by_session(session_id)
    if cell_store is not None:
        video = [r for r in recs if r.modality == "video"]
        turns = (_pair_turns([r for r in recs if r.modality != "video"])
                 + _video_turns(video, session_id, cell_store))
    else:
        turns = _pair_turns(recs)

    if covered > len(turns):        # the store was wiped or the watermark drifted -> reset (fault tolerance)
        covered = 0
        summary = ""
    tail = turns[covered:]

    # Trigger metric: the character count of the summary + (the tail minus the last keep_recent
    # turns); those last keep_recent turns do not count
    aged = tail[:-keep_recent] if len(tail) > keep_recent else []
    metric = len(summary) + sum(_turn_len(t) for t in aged)

    if aged and metric > cap:
        tgt = target or max(5000, cap // 10)
        new_summary = _compact(llm, summary, aged, tgt)
        # Still over cap after compacting: recompact with a harsher target, at most _MAX_RECOMPRESS
        # times, then hard-truncate if that still is not enough
        attempts = 1
        while len(new_summary) > cap and attempts < _MAX_RECOMPRESS:
            new_summary = _compact(llm, new_summary, [], max(300, tgt // 2))
            attempts += 1
        if len(new_summary) > cap:
            new_summary = new_summary[:cap] + " …[truncated]"
        summary = new_summary
        covered += len(aged)
        store.save(session_id, summary, covered)
        tail = turns[covered:]
        logger.info(f"session history compacted session={session_id} covered={covered} summary_len={len(summary)}")

    out: list[tuple[str, str]] = []
    if summary:
        out.append((_SUMMARY_HOLDER, summary))
    out.extend(_out_lines(tail))
    return out
