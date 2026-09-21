"""The bridge from video identity to the main memory pipeline (wiring line ownership).

Called after the commit at the end of a session: take the session's buffered screenplay lines
(draft.all_lines), apply the {chain_ref -> character_id} ownership mapping that commit delivered,
translate them into evidence carrying character ownership, and feed them to build_cell in one shot to
produce a memcell and its atoms. This makes "Bob said he likes hiking" land as an atom with
holder=Bob, converging on the same character across sessions and modalities.

Design (already agreed with the user):
- holder = the display name (the person's name when known, otherwise a stable short handle 人物#N;
  SW -> "user", ENV -> "env"), while the exact character id travels in source.character_id (so the
  rendering layer sees a clean speaker name and ownership/reference is not polluted by a bare ULID;
  the exact id persists in the payload).
- A clip is stored as raw media evidence (modality=video, content_ref=oss_key, no content_inline),
  and derived lines point back at it through source.raw_evidence_id; raw_clip entries do not go into
  build_cell's records (they have no text).
- One build_cell per session (merging the lines of all clips in time order = one recording, one
  episode).
- A clean standalone entry point: a future session_end consumption path can call it directly (it is
  not tied to any script).

- Bare ids inside line text are rewritten to display names too (matching mneme's
  _rewrite_label_mentions): clip processing already replaced the local id (P1) with the cast id (S1),
  and here that becomes the display name — otherwise the same person would go by three different
  names in memory.
- Non-speech lines (action) get an `(action)` prefix: transcript rendering is a uniform
  "holder: text", so without the marker an action description would be read by the episode LLM as
  words this person SPOKE.

Not in this round: per-person profile consolidation (profile_consolidate currently only produces the
user profile; that would be a separate new feature).
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from personos.identity.draft import DraftStore
from personos.identity.screenplay import ENV_WHO, WEARER_CAST_ID, rewrite_ids
from personos.identity.store import CharacterStore
from personos.models import EvidenceRecord
from personos.online.write_path import CellBuild, build_cell

# Scenario hint for first-person video, guiding episode/atom extraction to recognize environment,
# actions and multi-person dialogue (passed as build_cell's scenario).
VIDEO_SCENARIO = ("first-person wearable/robot video: the transcript below is a screenplay of one"
                  " recording — speaker lines are spoken words, plus action and environment"
                  " observations. 'user' is the camera wearer; other names are people in view."
                  " A speaker labelled 人物#N is a person whose name is not known yet; the first"
                  " lines of the transcript describe what each of them looks like.")

_ANON_PREFIX = "人物#"      # prefix of the stable short handle for an unnamed person


class _DisplayResolver:
    """Session cast -> (holder display name, character_id). An unnamed person gets the stable short
    handle 人物#N (consistent within this flush)."""

    def __init__(self, draft: DraftStore, char_store: CharacterStore,
                 by_chain: dict[str, str], session_id: str) -> None:
        self.draft = draft
        self.store = char_store
        self.by_chain = by_chain
        self.session_id = session_id
        self.wearer = char_store.session_wearer(session_id) or char_store.wearer_character() or ""
        self._handle: dict[str, str] = {}   # cid -> 人物#N (stable)

    def resolve(self, who: str) -> tuple[str, str]:
        if who == ENV_WHO:
            return "env", ""
        if who == WEARER_CAST_ID:
            return "user", self.wearer
        ref = self.draft.chain_ref(self.session_id, who)
        cid = self.by_chain.get(ref) or self.by_chain.get(self.draft.canonical_chain(ref)) or ""
        if not cid:
            return who, ""                      # fallback: with no ownership, use the session cast label (should not happen)
        names = self.store.names_for(cid)
        if names:
            return names[0], cid
        ch = self.store.get_character(cid)
        primary = (ch or {}).get("primary_name")
        if primary:
            return primary, cid
        return self._handle.setdefault(cid, f"{_ANON_PREFIX}{len(self._handle) + 1}"), cid

    def describe(self, cast_id: str) -> str:
        """A person's appearance description: prefer the one on this session's chain (the most
        recent), and fall back to the `appearance` in the persistent record (someone already known)."""
        ref = self.draft.canonical_chain(self.draft.chain_ref(self.session_id, cast_id))
        desc = ((self.draft.get_chain(ref) or {}).get("desc_text") or "").strip()
        if desc:
            return desc
        cid = self.by_chain.get(ref) or ""
        profile = ((self.store.get_character(cid) or {}).get("text_profile") or {}) if cid else {}
        return str(profile.get("appearance") or "").strip()

    def name_map(self, cast_ids) -> dict[str, str]:
        """cast id -> display name, used to rewrite bare ids inside line text.

        Note the whole table must be built before any rewriting: resolve hands out 人物#N to unnamed
        people **in call order**, so resolving while rewriting would give the same person a different
        number on different lines.
        """
        return {c: self.resolve(c)[0] for c in cast_ids if c}


def flush_session_to_memory(
    draft: DraftStore, by_chain: dict[str, str], *, session_id: str,
    char_store: CharacterStore, evidence_store: Any, cell_store: Any, atom_store: Any,
    chain_store: Any, llm: Any, embedder: Any, media_store: Any = None,
    clip_keys: Optional[dict[int, str]] = None, scenario: str = VIDEO_SCENARIO,
) -> Optional[CellBuild]:
    """Flush the session's screenplay lines into evidence carrying character ownership plus one
    memcell. Returns None when there are no lines."""
    lines = draft.all_lines(session_id)
    if not lines:
        logger.info(f"video memory flush: session={session_id} no screenplay lines, skipping")
        return None
    clip_keys = clip_keys or {}
    resolver = _DisplayResolver(draft, char_store, by_chain, session_id)

    # (1) Raw media evidence: one entry per clip that has a key (modality=video, no content_inline,
    # not part of records)
    raw_id: dict[int, str] = {}
    for ci in sorted({r["clip"] for r in lines}):
        key = clip_keys.get(ci)
        if not key:
            continue
        rec = EvidenceRecord(modality="video", content_ref=key, sha256=f"clip:{session_id}:{ci}",
                             source={"session_id": session_id, "clip_index": ci, "kind": "raw_clip"})
        raw_id[ci] = evidence_store.append(rec)

    # (2) Derived line evidence: holder = display name, with the exact character id in
    # source.character_id; collected in time order to feed build_cell.
    # Build the complete cast -> display name table first (including roster members who appear but
    # never speak: their ids can still show up inside someone else's action line), and only then
    # rewrite line by line — resolving as we go would make the 人物#N numbering of unnamed people
    # drift.
    # The order is order of first appearance, not set iteration order: the 人物#N numbering has to be
    # reproducible (otherwise re-running the same recording would make the same person 人物#1 one
    # time and 人物#2 the next). Roster members who never spoke go at the end, sorted by id as a
    # tiebreak.
    seen_order = list(dict.fromkeys(ln["who"] for ln in lines))
    # The full set of people cannot be just "those who spoke + the roster": someone may never open
    # their mouth for the whole recording (especially common for the wearer) and yet be mentioned in
    # someone else's action line ("... while SW films"). The by_chain that commit delivers is the
    # authoritative full set.
    others = ({r.split(":")[-1] for r in by_chain} | set(draft.load_roster(session_id))
              | {WEARER_CAST_ID, ENV_WHO}) - set(seen_order)
    all_casts = seen_order + sorted(others)
    names = resolver.name_map(all_casts)

    # (2.5) Intro lines for unnamed people: **listed once per person**, placed before the dialogue.
    # Without them, 人物#1 is just a hollow number in memory — the episode/atom extraction has no
    # idea who it refers to, there is no way to judge whether it is the same person across sessions,
    # and answering can only parrot the number back. People with names need no intro: the name is the
    # identity. Attached with holder=env (this is narration, not something anyone said).
    intro: list[EvidenceRecord] = []
    introduced: set[str] = set()
    for cast_id in all_casts:
        holder, cid = resolver.resolve(cast_id)
        if not holder.startswith(_ANON_PREFIX) or cid in introduced:
            continue      # two casts may have merged into one person after final review, so dedup by character_id
        desc = rewrite_ids(resolver.describe(cast_id), names)
        if not desc:
            continue
        introduced.add(cid)
        intro.append(EvidenceRecord(
            holder="env", content_inline=f"{holder} is {desc}", modality="video",
            source={"session_id": session_id, "kind": "cast_intro",
                    "cast_id": cast_id, "character_id": cid}))
    for rec in intro:
        evidence_store.append(rec)

    records: list[EvidenceRecord] = list(intro)
    for ln in lines:
        text = rewrite_ids((ln.get("text") or "").strip(), names)
        if not text:
            continue
        holder, cid = resolver.resolve(ln["who"])
        if (ln.get("kind") or "") == "action":
            text = f"(action) {text}"      # otherwise the action description gets read as words this person said
        ci = ln["clip"]
        rec = EvidenceRecord(
            holder=holder, content_inline=text, modality="video",
            content_ref=clip_keys.get(ci),
            source={"session_id": session_id, "clip_index": ci, "t0": ln["t0"], "t1": ln["t1"],
                    "kind": ln.get("kind"), "cast_id": ln["who"], "character_id": cid,
                    "raw_evidence_id": raw_id.get(ci)})
        evidence_store.append(rec)
        records.append(rec)
    if len(records) == len(intro):     # only intro lines, not a single line of dialogue or action -> nothing to record
        logger.info(f"video memory flush: session={session_id} no renderable lines, skipping build_cell")
        return None

    # (3) One build_cell: all the session's lines = one recording, one episode
    logger.info(f"video memory flush: session={session_id} lines={len(records) - len(intro)} "
                f"intro={len(intro)} raw_clips={len(raw_id)} → build_cell")
    return build_cell(llm, embedder, evidence_store, cell_store, atom_store, records,
                      session_id=session_id, chain_store=chain_store, scenario=scenario)
