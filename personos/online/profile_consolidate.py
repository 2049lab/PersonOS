"""Profile consolidation: a strong model emits a **patch** in one shot, with harness rejections and
over-cap rejections; short <-> long id mapping.

Single-stage (no digest): the input is the current profile (structured, with the long source ids
stripped) + the cells closed since the last published version (episode + atoms, labelled with the
short handles c1..cN) + today's date. The LLM only produces a patch (add / rewrite / drop, choosing
the band itself), profile_merge applies it, profile_harness validates it; there are two kinds of
rejection: invalid fields, or a band over cap. Once retries run out: a field-level failure keeps the
old version (returns None); a pure over-cap failure is persisted after the engine evicts the oldest
as a backstop.

Id mapping convention (see the llm-id-reference-short-long-mapping memory): the LLM only sees and
only writes the short handles c1..cN; validation works on the short-handle set; before merging they
are mapped back from short to the real cell_id.
"""

from __future__ import annotations

import json
from datetime import date

from loguru import logger

from .. import obs
from ..models import MemCell, MemoryAtom
from ..storage.profile_store import UserProfile
from .llm import ChatLLM, chat_json, with_scenario
from .profile_harness import validate_patch
from .profile_merge import BAND_CAPS, apply_patch, enforce_caps, over_cap

# Directive for caller-scenario injection (it only tunes attention; inventing profile content is
# forbidden) -- see llm.with_scenario
_SCEN_DIR_PROFILE = ("Weight traits and facts in the caller's focus area as higher-value to keep and "
                     "keep current; it never licenses inventing profile content unsupported by the "
                     "memory.")

_CONSOLIDATE_SYS = """# Role
You are the user-profile consolidator in a personal memory system. Given the CURRENT PROFILE
(structured JSON, may be empty), NEW MEMORY (recently closed topic cells: a narrative episode plus
extracted atoms each, labelled c1, c2, ...), and TODAY's date, output a PATCH that updates the profile.

The profile answers two things about the user:
- traits: what kind of person the user is (a stable portrait, inference allowed but marked)
- facts: noteworthy things that happened to / about the user, organized into recency bands

# Output: a PATCH, not a full rewrite (JSON only)
Emit ONLY what changes; anything you do not mention is preserved as-is. Never restate unchanged content.
{
  "traits": { "<domain>": {"text": "...", "status": "confirmed|inferred", "sources": ["c1"]},
              "<domain>": null },
  "facts": {
    "add":     [ {"band": "today|week|month|long", "text": "...", "sources": ["c1"]} ],
    "rewrite": [ {"id": "f_xxx", ...only the fields you change..., "band": "week"(optional, to move)} ],
    "drop":    [ "f_xxx" ]
  }
}
- sources MUST be short labels (c1, c2, ...) of cells in NEW MEMORY — never invent one, never write a long id.
- Do NOT write dates as a separate field — write dates INSIDE the text in natural language
  ("2026-09-14 had ramen"; "married in 2020"); resolve relative words to absolute dates.
- "traits": {"domain": null} clears a domain. Omit a domain to leave it untouched.

# traits — the portrait (PMO-16 domains in three layers; use these EXACT keys)
- L1 dispositional (what the person is like overall): personality, communication_style, social_style
- L2 characteristic adaptations (what they want / value / how they operate): occupation, goals, values,
  work_style, learning_style, tech_environment, lifestyle, health, finance
- L3 narrative identity (how they understand their own life): identity, location, family, interests
Rules:
- ONE concise sentence per domain (<= 500 chars). Prefer an if-then behavioral signature
  ("in technical talk prefers direct feedback; brief in small talk") over a flat trait ("is direct").
- REWRITE-INTEGRATE, do not accumulate: when a domain changes, rewrite the whole sentence folding
  old + new into one smooth sentence; never append fragments or let it grow longer and longer.
- status: confirmed = user stated it; inferred = you deduced from behavior (allowed, but mark it).
- A domain no dialogue exposes stays null — never fabricate to fill a slot.

# facts — recency bands (YOU place each fact in a band; the engine only enforces per-band caps)
Band responsibilities:
- today  (max 1): TODAY's facts only, integrated into ONE item.
- week   (max 3): facts from roughly the last 7 days.
- month  (max 5): facts from roughly the last 30 days.
- long   (max 21): only genuinely ENDURING facts (identity, lasting preferences, stable relationships,
  values). Put a fact here by "band":"long".
Near bands may hold plain objective facts (a log); long holds only what stays true.

YOU maintain the bands as of TODAY (there is no automatic date sorting):
- Roll aged facts DOWN yourself: a fact that was "today" yesterday is no longer today's — move it to
  week (rewrite with "band":"week"), and free the today slot for today's new content. Likewise week→month.
- Never keep extending a past day's fact to stuff today's events into it — create a NEW fact for today,
  and let the older one roll down (or drop / promote it).
- A single fact MAY span multiple days as a theme ("worked Mon–Fri on project X" = one week fact) —
  the band is your judgement of recency/importance, not a per-day limit.
- If a band would exceed its cap, MERGE related items, DROP the least important, or PROMOTE the enduring
  one to long — do not overflow.

Fact writing discipline (mirror atom extraction):
- Keep every qualifier (when / where / with whom / how / how often / until when); proper nouns and
  numbers verbatim; third person; dates written in the text (dual time: relative + absolute).
- One fact may INTEGRATE several atoms sharing one bounding condition (e.g. same day).
- Facts are observed, not guessed — record what was said.

# Language
Write trait/fact text in the SAME language as the source dialogue.

# Output JSON only (no prose, no code fences)."""


def _sanitized_profile_json(current: UserProfile | None) -> str:
    """The current profile as the LLM sees it: sources are stripped (they are real long ids, which the
    LLM neither needs nor should copy), while id / text / band are kept."""
    if current is None:
        return "{}"
    d = current.model_dump()
    for t in d.get("traits", {}).values():
        if t:
            t.pop("sources", None)
    for band in d.get("facts", {}).values():
        for f in band:
            f.pop("sources", None)
    return json.dumps(d, ensure_ascii=False)


def _render_input(current: UserProfile | None, cells: list[MemCell],
                  atoms_by_cell: dict[str, list[MemoryAtom]], today: date) -> tuple[str, dict[str, str]]:
    """Render the user message and return the {short handle: real cell_id} mapping (new cells are
    labelled c1..cN)."""
    cell_map: dict[str, str] = {}
    blocks = []
    for i, c in enumerate(cells, 1):
        label = f"c{i}"
        cell_map[label] = c.id
        lines = [f"━━━ {label} ━━━ topic: {c.topic}", f"episode: {c.episode}"]
        atoms = atoms_by_cell.get(c.id, [])
        if atoms:
            lines.append("atoms:")
            lines.extend(f"- {a.text}" for a in atoms)
        blocks.append("\n".join(lines))
    material = "\n\n".join(blocks) if blocks else "(none)"
    msg = (f"TODAY: {today.isoformat()}\n\n"
           f"CURRENT PROFILE (JSON; {{}} = no profile yet):\n{_sanitized_profile_json(current)}\n\n"
           f"NEW MEMORY (topic cells closed since last consolidation; cite these labels as sources):\n"
           f"{material}")
    return msg, cell_map


def _remap_sources(patch: dict, cell_map: dict[str, str]) -> dict:
    """Map short handles back to real cell_ids (validation already guaranteed every handle is in the
    set). Returns a new patch; the argument is not mutated."""
    def mp(src):
        return [cell_map[s] for s in (src or []) if s in cell_map]

    out = json.loads(json.dumps(patch))          # deep copy
    for t in (out.get("traits") or {}).values():
        if isinstance(t, dict) and "sources" in t:
            t["sources"] = mp(t["sources"])
    facts = out.get("facts") or {}
    for a in facts.get("add") or []:
        if isinstance(a, dict) and "sources" in a:
            a["sources"] = mp(a["sources"])
    for rw in facts.get("rewrite") or []:
        if isinstance(rw, dict) and "sources" in rw:
            rw["sources"] = mp(rw["sources"])
    return out


def _retry_msg(errs: list[str]) -> str:
    bullet = "\n".join(f"- {e}" for e in errs[:20])
    return ("你上一版补丁有以下问题,请逐条修正后重新**只输出 JSON 补丁本体**"
            f"(不要解释、不要代码块围栏):\n{bullet}")


def consolidate(llm: ChatLLM, *, current: UserProfile | None, cells: list[MemCell],
                atoms_by_cell: dict[str, list[MemoryAtom]], today: date,
                max_retries: int = 2, max_tokens: int = 4096,
                scenario: str = "") -> UserProfile | None:
    """Run one consolidation. No new cells -> None. Still failing after the field-level rejections run
    out -> None (the old version is kept); a pure over-cap failure that will not converge -> return
    after the engine evicts the oldest as a backstop.
    scenario: the caller's scenario description (may be empty) — it biases traits/facts toward the
    domains the caller cares about."""
    if not cells:
        return None
    today_str = today.isoformat()
    user_msg, cell_map = _render_input(current, cells, atoms_by_cell, today)
    valid = set(cell_map.keys())
    sys = with_scenario(_CONSOLIDATE_SYS, "# Output: a PATCH, not a full rewrite (JSON only)",
                        scenario, _SCEN_DIR_PROFILE)
    messages = [{"role": "system", "content": sys},
                {"role": "user", "content": user_msg}]
    last_valid: UserProfile | None = None         # a candidate with valid fields but over cap (kept for the backstop)

    for attempt in range(max_retries + 1):
        raw = ""
        try:
            patch, raw = chat_json(llm, messages, max_tokens=max_tokens, temperature=0.2,
                                   stage="profile_consolidate")
            errs = validate_patch(patch, valid_cell_ids=valid)
        except ValueError as e:                   # the output was not valid JSON
            raw = getattr(e, "raw", "")
            patch, errs = None, ["你的输出不是合法 JSON,请只输出 JSON 补丁本体"]

        if patch is not None and not errs:
            merged = apply_patch(current, _remap_sources(patch, cell_map),
                                 today_str=today_str, evict=False)
            over = over_cap(merged)
            if not over:
                logger.info(f"consolidate succeeded attempt={attempt + 1} cells={len(cells)}")
                return merged
            last_valid = merged                    # fields are valid, only the cap is exceeded
            errs = [f"{b} 带现有 {n} 条,超上限 {BAND_CAPS[b]};请合并/删除/升 long 收敛到上限内"
                    for b, n in over.items()]

        if attempt < max_retries:
            logger.warning(f"consolidate rejected attempt={attempt + 1} errs={errs[:3]}")
            messages = messages[:2] + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _retry_msg(errs)},
            ]

    if last_valid is not None:                     # pure over-cap that will not converge -> publish after the engine's backstop evicts the oldest
        enforce_caps(last_valid)
        logger.warning(f"consolidate over-cap rejections did not converge, publishing after the engine evicted the oldest as a backstop (cells={len(cells)})")
        return last_valid
    logger.warning(f"consolidate still failing field validation after {max_retries} rejections, keeping the previous version (cells={len(cells)})")
    return None


# -- Trigger annealing + per-user orchestration (used by the runtime wiring; split out to make it
# unit-testable) --

def anneal_step(version_count: int) -> int:
    """The annealing step size (by number of published versions): 1, 2, 2, 3, 4, 5, ... capped at 5.
    On a cold start this gets the first version into shape as early as possible."""
    ramp = (1, 2, 2, 3, 4, 5)
    return ramp[version_count] if version_count < len(ramp) else 5


def should_consolidate(*, n_new: int, ep_chars: int, version_count: int,
                       ep_chars_trigger: int) -> bool:
    """Trigger decision: fires when either the number of new cells reaches the annealing step size or
    the accumulated episode character count reaches its ceiling."""
    return n_new >= anneal_step(version_count) or ep_chars >= ep_chars_trigger


def run_user_consolidation(llm: ChatLLM, *, cells_store, atoms_store, profile_store,
                           today: date, scenario: str = "") -> int | None:
    """Consolidation orchestration for one user: read the cells + atoms after the cursor ->
    consolidate -> publish a version. Returns the new version number or None.

    Idempotent: what it reads is "the cells since the last published version", so if a single-flight
    lock turns it away or a trigger is missed, the next trigger heals it (it still reads all the new
    cells).
    """
    cur = profile_store.current()
    cursor = cur.up_to_cell_id if cur else ""
    cells = cells_store.cells_after(cursor)
    if not cells:
        return None
    atoms_by_cell = {c.id: atoms_store.list_by_cell(c.id) for c in cells}
    _inp = "; ".join(c.topic for c in cells if c.topic)[:500] or f"{len(cells)} new cells"
    with obs.root_span("profile.consolidate", user_id=getattr(profile_store, "user_id", None),
                       session_id="profile", input=_inp, metadata={"n_cells": len(cells)}):
        merged = consolidate(llm, current=cur.profile if cur else None,
                             cells=cells, atoms_by_cell=atoms_by_cell, today=today,
                             scenario=scenario)
        if merged is None:
            return None
        version = profile_store.save_version(merged, up_to_cell_id=cells[-1].id)
        traits_detail = "\n".join(f"    {dom}: {t.text!r} ({t.status})"
                                  for dom, t in merged.traits.items() if t)
        facts_detail = "\n".join(
            f"    [{band}] " + " | ".join(f.text for f in merged.facts.get(band, []))
            for band in merged.facts)
        logger.info(f"profile published v{version} user={getattr(profile_store, 'user_id', '?')} "
                    f"cells={len(cells)}\n  ── traits ──\n{traits_detail}\n  ── facts ──\n{facts_detail}")
        return version
