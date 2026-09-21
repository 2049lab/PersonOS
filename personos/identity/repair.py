"""**Unified repair** of physical contradictions in a screenplay: detect them all at
once, hand them back to the model in one go, and degrade conservatively if that
still does not clear them.

Why this layer exists: a multimodal LLM hallucinates, and code judges
**physically impossible** errors — one person taken for two characters, one person
in two places at once — far more reliably than the model judging itself. Having
detected them we do not simply drop the data; we feed **all** the contradictions
back together with the model's own previous output and ask it to fix them. If it
is plainly wrong, it should get to re-decide.

One deliberate simplification: this could be split into two repair loops, with
cast-type rules before harvest and voice-type rules after it (a voice repair
prompt could then use the extracted face photos). We **merge them into a single
pass, entirely before harvest**, because:

- rules 1/2/3/5/7 are all logical contradictions rather than judgements about a
  face, so the video plus the roster cards is enough;
- it is one call instead of two, and each call re-sends the video, which costs
  around 100s on our deployment — saving one is worth it;
- a bad nomination gets fixed **before any face is extracted**, so a wrong face
  never gets the chance to enter the probability cloud.

The flow, over at most two rounds:

    detect -> build [previous output + contradiction list] -> re-run -> merge
    (with a referential-integrity check) -> re-detect -> accept if clean; if
    contradictions remain, **adopt the partial improvement as the new baseline**
    and try another round carrying what is left -> still failing after two rounds
    -> degrade conservatively, by subtraction only

Degrading always errs toward having less rather than having something wrong: drop
the suspect nomination, voice range or name, and never invent an attribution. The
cost of dropping something is bounded and recoverable, since a later clip will
nominate the same person again, whereas a wrong attribution learned into the
probability cloud is irreversible.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Optional

from loguru import logger

from personos.identity.inspect import (
    DEGRADE_ONLY, REPAIRABLE, Violation, inspect_script,
)
from personos.identity.screenplay import (
    ENV_WHO, WEARER_CAST_ID, CastDecl, ClipScript, Nomination, VoiceRange,
    parse_clip_output,
)

MAX_REPAIR_ATTEMPTS = 2
_MIN_VOICE_SEC = 0.4       # kept in step with harvest._MIN_VOICE_SEC and inspect

# Three settings: degrade = detect, repair and degrade (the default);
# detect = detect and record only, leaving the screenplay untouched; off = disabled.
MODE = os.environ.get("PERSONOS_IDENTITY_GUARD", "degrade").strip().lower()

_HEADER = """# Role
You are fixing **physical contradictions** in a screenplay you produced for this video clip.
Each contradiction below is impossible in the real world — one person cannot be two people,
one person cannot be in two places at the same instant, and two people cannot both own the
same stretch of speech. So at least one of your records is wrong.

# Task
Re-output ONLY the `casts`, `noms`, `voices` and `conts` arrays, corrected.
- Do NOT re-output `lines` — the transcript itself is kept as-is and shown to you for context.
- Every id referenced by the kept lines and by your voices MUST still exist in `casts`.
- Watch the clip again before deciding; prefer dropping a record you are unsure about over
  guessing — a missing record is recoverable, a wrong attribution is not.

# Output
ONLY the JSON object, no fences, no prose:
{"casts": [...], "noms": [...], "voices": [...], "conts": [...]}
Same field rules as your previous output."""


def _violation_block(violations: list[Violation]) -> str:
    return "\n".join(f"- [{v.rule}] {v.detail}" for v in violations)


def _render_cast_records(script: ClipScript) -> str:
    """Feed the model's own previous casts/noms/voices/conts back verbatim, so it
    edits its output rather than rewriting it from scratch.
    """
    def _cast(c: CastDecl) -> dict:
        return {"id": c.local_id, "wearer": c.is_wearer, "name": c.name or "",
                "name_evidence": c.name_evidence, "desc": c.desc}
    payload = {
        "casts": [_cast(c) for c in script.casts],
        "noms": [{"id": n.local_id, "t": round(n.t, 2), "pos": n.pos, "desc": n.desc}
                 for n in script.nominations],
        "voices": [{"id": v.local_id, "t0": round(v.t0, 2), "t1": round(v.t1, 2)}
                   for v in script.voice_ranges],
        "conts": [{"id": k, "prev": v, "evidence": script.cont_evidence.get(k, "")}
                  for k, v in (script.cont or {}).items()],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _render_lines_context(script: ClipScript, cap: int = 40) -> str:
    return "\n".join(f"[{l.t0:.1f}-{l.t1:.1f}] {l.who} ({l.kind}): {l.text}"
                     for l in script.lines[:cap])


def build_repair_prompt(script: ClipScript, violations: list[Violation],
                        roster_cards: str = "") -> str:
    """Assemble [rule header + roster + previous output + kept lines + contradiction
    list] — every problem handed over at once.
    """
    blocks = [_HEADER]
    if roster_cards:
        blocks.append("ROSTER (people already seen earlier in this recording):\n" + roster_cards)
    blocks.append("YOUR PREVIOUS RECORDS:\n" + _render_cast_records(script))
    blocks.append("KEPT LINE RECORDS (context only — do NOT re-output):\n"
                  + _render_lines_context(script))
    blocks.append("CONTRADICTIONS TO FIX:\n" + _violation_block(violations))
    blocks.append("Begin corrected JSON now.")
    return "\n\n".join(blocks)


def merge_repair(script: ClipScript, raw: str,
                 duration_sec: Optional[float]) -> Optional[ClipScript]:
    """Merge the repair output back into the original screenplay, replacing only
    casts/noms/voices/conts and keeping lines as they were.

    **Referential integrity check**: every id referenced by the kept lines and by
    the new voices must still exist in casts. If one does not, the whole repair is
    voided and None is returned — a line's attribution must not dangle, so we would
    rather fall back to the previous version.
    """
    # Reuse the main parser: the repair output has the same schema as a screenplay
    # minus lines, so parse it and then put the original lines back.
    patched = parse_clip_output(raw, duration_sec=duration_sec)
    if not patched.parsed_ok or not patched.casts:
        return None
    ids = {c.local_id for c in patched.casts} | {WEARER_CAST_ID, ENV_WHO}
    for line in script.lines:
        if line.who not in ids:
            logger.warning(f"repair voided: a kept line references nonexistent id {line.who!r}")
            return None
    for vr in patched.voice_ranges:
        if vr.local_id not in ids:
            logger.warning(f"repair voided: a voice range references nonexistent id {vr.local_id!r}")
            return None
    return ClipScript(
        casts=patched.casts, lines=list(script.lines), nominations=patched.nominations,
        voice_ranges=patched.voice_ranges, cont=patched.cont,
        cont_evidence=patched.cont_evidence,
        issues=[*script.issues, *patched.issues], parsed_ok=True, raw=raw)


def degrade(script: ClipScript, violations: list[Violation]) -> ClipScript:
    """Conservative degrade: **subtraction only**. Drop the suspect records and never
    invent an attribution.

    The cost of dropping something is bounded and recoverable — a later clip will
    nominate the same person again, and the name is still on the roster — whereas a
    wrong attribution learned into the probability cloud is irreversible. So the
    bias is always toward having less rather than having something wrong.
    """
    casts, noms = list(script.casts), list(script.nominations)
    voices, cont = list(script.voice_ranges), dict(script.cont or {})
    hit = {v.rule for v in violations}
    notes: list[str] = []

    if "duplicate_cast" in hit:                       # the same id declared repeatedly -> keep only the first
        seen, kept = set(), []
        for c in casts:
            if c.local_id in seen:
                notes.append(f"dropped duplicate declaration {c.local_id}")
                continue
            seen.add(c.local_id); kept.append(c)
        casts = kept

    if "cont_conflict" in hit:                        # break the implied continuation and let arbitration decide from the assets
        for v in violations:
            if v.rule != "cont_conflict":
                continue
            for cid in v.cast_ids:
                if cont.pop(cid, None) is not None:
                    notes.append(f"broke continuation {cid}")

    drop_t: dict[str, set[float]] = {}                # nominations that split a person in two, or nominate the wearer -> drop them
    for v in violations:
        if v.rule in ("nom_position_conflict", "wearer_visible"):
            for cid in v.cast_ids:
                drop_t.setdefault(cid, set()).update(v.times)
    if drop_t:
        before = len(noms)
        noms = [n for n in noms
                if not any(abs(n.t - t) < 1e-6 for t in drop_t.get(n.local_id, ()))]
        if len(noms) != before:
            notes.append(f"dropped {before - len(noms)} nominations")

    for v in violations:                              # overlapping voices -> neither side keeps the overlapping span
        if v.rule != "voice_overlap" or len(v.times) != 2:
            continue
        lo, hi = v.times
        cut: list[VoiceRange] = []
        for r in voices:
            if r.local_id not in v.cast_ids or r.t1 <= lo or r.t0 >= hi:
                cut.append(r)
                continue
            for a, b in ((r.t0, lo), (hi, r.t1)):     # keep only the leftovers that do not overlap and are long enough
                if b - a >= _MIN_VOICE_SEC:
                    cut.append(replace(r, t0=a, t1=b))
            notes.append(f"trimmed voice range {r.local_id}[{r.t0:.1f},{r.t1:.1f}]")
        voices = cut

    for v in violations:                              # a name with no support in the speech -> strip it; the roster still has it
        if v.rule not in DEGRADE_ONLY:
            continue
        for cid in v.cast_ids:
            casts = [replace(c, name=None, name_evidence="none") if c.local_id == cid else c
                     for c in casts]
            notes.append(f"stripped name from {cid}")

    return ClipScript(
        casts=casts, lines=list(script.lines), nominations=noms, voice_ranges=voices,
        cont=cont, cont_evidence=dict(script.cont_evidence),
        issues=[*script.issues, *(f"degraded: {n}" for n in notes)],
        parsed_ok=script.parsed_ok, raw=script.raw)


def enforce(script: ClipScript, *, omni: Any = None, clip_url: str = "",
            roster_cards: str = "", duration_sec: Optional[float] = None,
            session_id: str = "", clip_index: int = 0) -> tuple[ClipScript, dict]:
    """Detect everything, repair in one pass (at most two rounds), then degrade
    conservatively. Returns (screenplay, report).

    The report fields let the caller emit structured logs and statistics: how many
    were found, fixed and left over, counted by rule, and how many repair rounds
    ran. Nothing is ever raised — a broken guard layer must not take down the whole
    clip, and the worst case degrades to "no guarding".
    """
    rep: dict[str, Any] = {"mode": MODE, "found": {}, "attempts": 0,
                           "remaining": {}, "degraded": False}
    if MODE == "off":
        return script, rep
    try:
        return _enforce(script, rep, omni=omni, clip_url=clip_url, roster_cards=roster_cards,
                        duration_sec=duration_sec)
    except Exception as e:  # noqa: BLE001  if the guard layer itself breaks, degrade to "no guarding" rather than losing the clip
        logger.exception(f"consistency guard failed, leaving this clip unguarded session={session_id} clip={clip_index}: {e}")
        rep["error"] = f"{type(e).__name__}: {e}"
        return script, rep


def _enforce(script: ClipScript, rep: dict, *, omni: Any, clip_url: str,
             roster_cards: str, duration_sec: Optional[float]) -> tuple[ClipScript, dict]:
    violations = inspect_script(script)
    rep["found"] = _by_rule(violations)
    if not violations:
        return script, rep
    if MODE == "detect":       # record only, leave the screenplay alone: for collecting trigger rates after a rollout
        rep["remaining"] = rep["found"]
        return script, rep

    repairable = [v for v in violations if v.rule in REPAIRABLE]
    if repairable and omni is not None and clip_url:
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            rep["attempts"] = attempt
            try:
                prompt = build_repair_prompt(script, repairable, roster_cards)
                raw = omni.chat(prompt, video_url=clip_url, max_tokens=8000, temperature=0.0)
                patched = merge_repair(script, raw, duration_sec)
            except Exception as e:  # noqa: BLE001  a failed repair degrades instead of taking down the clip
                logger.warning(f"screenplay repair round {attempt} failed, falling back to degrade: {e}")
                break
            if patched is None:
                continue
            left = inspect_script(patched)
            # Adopt even a partial improvement as the new baseline — referential
            # integrity has already been checked — and try another round carrying
            # whatever contradictions are left.
            script, violations = patched, left
            repairable = [v for v in left if v.rule in REPAIRABLE]
            if not repairable:
                break

    if violations:
        script = degrade(script, violations)
        rep["degraded"] = True
        rep["remaining"] = _by_rule(violations)
    return script, rep


def _by_rule(violations: list[Violation]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in violations:
        out[v.rule] = out.get(v.rule, 0) + 1
    return out
