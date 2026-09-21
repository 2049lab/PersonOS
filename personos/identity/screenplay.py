"""The single-pass screenplay protocol: a JSON-schema prompt plus parsing and validation.

Dialogue, action and environment all come back in one pass.

Why JSON rather than fixed-width pipe-delimited text: fixed-width text identifies
fields by position, so the moment the model drops a segment (say it omits kind) or
writes a bare numeric id, the whole row shifts and the parser has to patch it up.
JSON names its keys explicitly, the model follows it closely, json.loads is a
stable parse, and we almost never need a fallback. The principle is to get the
multimodal model to emit something correct directly rather than patching output
after the fact.

Design choices worth calling out:

- **one pass**: a single prompt produces dialogue, action and environment (the
  three line kinds) together, instead of three separate passes;
- **salience gating**: a cast entry is created only for load-bearing people —
  those who speak, interact with the wearer, are addressed by name, or perform a
  notable action. Passers-by go into environment lines instead;
- **horizontal-thirds pos**: a nomination's pos is left/center/right. People in
  frame separate mostly left-to-right, so there is no vertical axis. After the box
  is picked, asset attribution re-checks it for agreement — if the chosen face's
  actual third does not match the nomination we refuse rather than guess. See
  harvest.pick_face.

Output is a single JSON object {casts, lines, noms, voices, conts}; the fields are
documented by the schema inside build_clip_prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

WEARER_CAST_ID = "SW"
ENV_WHO = "ENV"
NAME_EVIDENCE_LEVELS = ("none", "spoken", "visible_text", "self_introduction",
                        "explicit_dialogue", "introduction")
LINE_KINDS = ("speech", "action", "environment")

def rewrite_ids(text: str, mapping: dict[str, str]) -> str:
    """Replace **bare person ids inside line text** with the names given by mapping.

    Why this is necessary: when the screenplay model writes action and environment
    lines it addresses people by id right in the prose — "P1 enters holding a
    basketball", "where P2 is seated". Rewriting attribution only touches the
    holder, so those ids leak straight through evidence -> episode -> atom, and one
    person ends up with three names in memory: whatever the holder calls them,
    "P1" in the text, and "Alice" in someone else's mouth. Retrieval and answering
    can then no longer tell that these are the same person.

    It happens in two stages, each where the information is complete: while
    processing a clip, local id (P1) -> session cast id (S1), which is what
    cast_map gives us; at end-of-session flush, cast id (S1) -> display name,
    which is what final adjudication gives us.

    Only keys actually present in mapping are replaced, and word boundaries are
    required so ordinary words are left alone. Longer keys go first, so that P1
    cannot chop P12 in half.
    """
    if not text or not mapping:
        return text
    keys = sorted((k for k in mapping if k), key=len, reverse=True)
    if not keys:
        return text
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pat.sub(lambda m: mapping[m.group(1)], text)


_LOCAL_ID_RE = re.compile(r"^P[0-9]+$|^S[0-9]+$|^SW$")
_PREV_RE = re.compile(r"^S[0-9]+$|^SW$|^none$")
_POS_VALUES = ("left", "center", "right")   # the horizontal thirds


# ── Data structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CastDecl:
    local_id: str
    is_wearer: bool = False
    name: str | None = None
    name_evidence: str = "none"
    desc: str = ""


@dataclass(frozen=True)
class ClipLine:
    t0: float
    t1: float
    who: str                      # local_id | "SW" | "ENV"
    kind: str                     # speech | action | environment
    text: str


@dataclass(frozen=True)
class Nomination:
    local_id: str
    t: float
    desc: str = ""
    pos: str = ""                 # horizontal third left/center/right, used to disambiguate between faces; empty means do not pick by position


@dataclass(frozen=True)
class VoiceRange:
    local_id: str
    t0: float
    t1: float


@dataclass
class ClipScript:
    casts: list[CastDecl] = field(default_factory=list)
    lines: list[ClipLine] = field(default_factory=list)
    nominations: list[Nomination] = field(default_factory=list)
    voice_ranges: list[VoiceRange] = field(default_factory=list)
    cont: dict[str, str] = field(default_factory=dict)
    cont_evidence: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    parsed_ok: bool = False       # the JSON parsed; this replaces the old pipe protocol's saw_end
    raw: str = ""
    cast_map: dict[str, str] = field(default_factory=dict)  # local_id -> session cast, filled in by the mapping side

    def cast_decl(self, local_id: str) -> CastDecl | None:
        return next((c for c in self.casts if c.local_id == local_id), None)

    def session_cast_ids(self) -> list[str]:
        """The session cast ids appearing in this clip.

        These are the cast_map values, deduplicated while keeping order, and are
        available once the mapping side has filled cast_map in.
        """
        seen: list[str] = []
        for cast_id in self.cast_map.values():
            if cast_id not in seen:
                seen.append(cast_id)
        return seen


# ── The prompt: the single-pass screenplay JSON schema ───────────────

DESC_SPEC = ("rich identifying appearance — hair (color/length/style); face shape & notable "
             "features (glasses, facial hair, marks); build/height/posture; skin tone; age "
             "impression; clothing with colors. Detailed enough to pick this person out of a "
             "small crowd from the description alone")

_NAME_RULES = (
    'A "name" is reported only if evidenced in THIS clip (spoken in dialogue, self-introduction, '
    'or visible text) — never guessed, never copied from the ROSTER. A spoken name belongs to the '
    'person ADDRESSED, not the speaker ("Hi Alice" → name on the person spoken TO, '
    'name_evidence="explicit_dialogue"); a name attaches to its own speaker only via explicit '
    'self-introduction "I\'m X" (self_introduction); introducing someone present ("This is X") → '
    'name on the person INTRODUCED (introduction). Otherwise name="".')


def build_clip_prompt(*, scene_setting: str = "",
                      roster_cards: list[dict[str, Any]] | None = None) -> tuple[str, list[str]]:
    """Build the single-pass screenplay JSON prompt.

    Returns (prompt, list of roster image b64).

    scene_setting is the caller's framing of the scene (first-person glasses,
    a robot, a documentary...), so the model reads the footage the right way.

    roster_cards are the casts seen earlier in this session, with names,
    appearance and reference images. They are reference material for judging
    continuation, not conclusions.
    """
    scene = scene_setting.strip() or "first-person wearable-camera"
    roster_block, images = _render_roster(roster_cards or [])
    header = f"""You are watching ONE clip of a longer {scene} recording. Read it faithfully and
return a single JSON object describing the screenplay and the people in it.

Output ONLY the JSON object — no markdown fences, no prose. Schema (all arrays; omit an array if empty):
{{
  "casts": [{{"id": "P1", "wearer": false, "name": "", "name_evidence": "none", "desc": "..."}}],
  "lines": [{{"t0": 0.0, "t1": 5.0, "who": "P1", "kind": "speech", "text": "..."}}],
  "noms":  [{{"id": "P1", "t": 6.0, "pos": "center", "desc": "..."}}],
  "voices":[{{"id": "P1", "t0": 7.0, "t1": 8.0}}],
  "conts": [{{"id": "P1", "prev": "none", "evidence": "..."}}]
}}

FIELD RULES:
- id: for a NEW person use "P1","P2",...; "SW" for the wearer (voice/hands BEHIND the camera, never visible). **If this person is a ROSTER member you have seen earlier in THIS recording, you MAY reuse their roster id "S#" directly as their id here — reusing the S# id itself declares the continuation.** ids are "P#" / "S#" / "SW" strings — never bare numbers, never names.
- casts: one entry per LOAD-BEARING person only (SALIENCE). A person qualifies if they speak, interact with the wearer, are addressed by name, or perform a notable action. Do NOT create a cast for background people / passers-by / a crowd — describe those in an environment line instead ("a busy street with many pedestrians"). A person with a missing modality (off-screen voice, or silent-but-visible) still gets a cast entry. If the wearer (SW) speaks or performs an action, include an "SW" cast entry (wearer=true) as well. name_evidence ∈ {{explicit_dialogue, self_introduction, introduction, visible_text, spoken, none}}. desc: {DESC_SPEC}.
- {_NAME_RULES}
- lines: time-ordered, seconds within THIS clip. kind ∈ {{speech, action, environment}} (REQUIRED per line):
    speech = verbatim words in the original spoken language, who = the speaker id;
    action = a notable visible action (who did what to what), who = the actor id or "SW";
    environment = scene/layout, on-screen text/signs, notable objects & locations, salient non-speech sounds, who = "ENV".
  KEEP IT CONCISE AND MEMORY-WORTHY: record speech verbatim (it carries the meaning), but for action/environment
  write only what matters for remembering this scene — one line per meaningful action or notable scene fact.
  Do NOT narrate transient/repetitive micro-detail (every basketball bounce, each footstep, every "okay", minor
  gestures). Describe clearly, not exhaustively — fewer, cleaner lines are better than blow-by-blow narration.
- noms: for EVERY visible cast member, 1-3 moments where they are most recognizable (build/body/clothing/face; face need not be clear; STRONGLY prefer moments where the person is ALONE or clearly separated from others — those give the cleanest reference). pos is REQUIRED — the person's HORIZONTAL position in the frame at that moment, one of "left" / "center" / "right". When two people are close together, pick moments where they are on clearly different sides. This is how co-appearing people are told apart. Never nominate "SW".
- voices: only ranges where that person speaks completely ALONE (no overlap); missing is better than dirty.
- conts: CONTINUITY MATTERS — check EVERY person against the ROSTER. If a person continues a ROSTER member seen earlier in THIS recording, you MUST continue them: either reuse that "S#" as their id (see id rule), OR add a cont with prev="S#". Only a genuinely new or uncertain person uses prev="none"; when unsure, prev="none". Do NOT mint a new P# for someone already on the roster."""

    return "\n\n".join([header, roster_block, "Return the JSON object now."]), images


def _render_roster(cards: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not cards:
        return "ROSTER: (empty — this is the first clip of the recording)", []
    images: list[str] = []
    rows = ["ROSTER (people seen earlier in THIS recording — reference material, NOT conclusions):"]
    for c in cards:
        rows.append(f"- cast {c.get('cast_id', '?')}:"
                    + (f" name={c['name']}" if c.get("name") else "")
                    + (f" appearance={c['desc']}" if c.get("desc") else ""))
        if c.get("face_b64"):
            images.append(c["face_b64"]); rows[-1] += f" [face=image #{len(images)}]"
        if c.get("body_b64"):
            images.append(c["body_b64"]); rows[-1] += f" [body=image #{len(images)}]"
    return "\n".join(rows), images


# ── Parsing: validate the JSON, never repair its formatting ──────────

def _extract_json(raw: str) -> str:
    """Strip markdown fences and take from the first { to the last }.

    This tolerates the model occasionally adding fences or a prefix/suffix.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:] if s[:4].lower() == "json" else s
        s = s.strip()
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if 0 <= i < j else s


def _fnum(v: Any) -> float:
    return float(v)


def _clamp(value: float, duration: float | None) -> float:
    return max(0.0, value) if duration is None else min(max(0.0, value), duration)


def parse_clip_output(raw: str, *, duration_sec: float | None = None) -> ClipScript:
    """Parse the model's JSON screenplay.

    A bad record is recorded in issues and skipped, so one failure does not
    cascade. If the JSON as a whole is bad, parsed_ok stays False.
    """
    script = ClipScript(raw=raw)
    try:
        data = json.loads(_extract_json(raw))
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except Exception as exc:  # noqa: BLE001
        script.issues.append(f"JSON parse failed: {type(exc).__name__}: {exc}")
        return script
    script.parsed_ok = True

    for i, c in enumerate(data.get("casts") or []):
        try:
            _add_cast(script, c)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"cast[{i}]: {exc}")
    declared = {c.local_id for c in script.casts} | {WEARER_CAST_ID}  # SW is the reserved wearer id and is always legal
    for i, ln in enumerate(data.get("lines") or []):
        try:
            _add_line(script, ln, duration_sec, declared)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"line[{i}]: {exc}")
    for i, n in enumerate(data.get("noms") or []):
        try:
            _add_nom(script, n, duration_sec, declared)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"nom[{i}]: {exc}")
    for i, v in enumerate(data.get("voices") or []):
        try:
            _add_voice(script, v, duration_sec, declared)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"voice[{i}]: {exc}")
    for i, ct in enumerate(data.get("conts") or []):
        try:
            _add_cont(script, ct, declared)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"cont[{i}]: {exc}")
    # SW (the wearer) was referenced but never declared as a cast, so add one. SW is
    # a structurally reserved id, which makes this a completion rather than a
    # format patch.
    if not script.cast_decl(WEARER_CAST_ID) and (
            any(l.who == WEARER_CAST_ID for l in script.lines)
            or any(v.local_id == WEARER_CAST_ID for v in script.voice_ranges)):
        script.casts.insert(0, CastDecl(local_id=WEARER_CAST_ID, is_wearer=True,
                                        desc="wearer behind the camera"))
    if not script.casts and script.lines:
        script.issues.append("no cast declared but lines present")
    return script


def _add_cast(script: ClipScript, c: dict) -> None:
    local_id = str(c.get("id", "")).strip()
    if not _LOCAL_ID_RE.fullmatch(local_id):
        raise ValueError(f"bad cast id {local_id!r}")
    name = (c.get("name") or "").strip()
    if name.lower() in {"-", "(unknown)", "unknown", "none"}:
        name = ""
    ev = str(c.get("name_evidence", "none")).lower()
    if ev not in NAME_EVIDENCE_LEVELS:
        ev = "none"
    is_wearer = bool(c.get("wearer")) or local_id == WEARER_CAST_ID
    script.casts.append(CastDecl(local_id=local_id, is_wearer=is_wearer,
                                 name=name or None, name_evidence=ev if name else "none",
                                 desc=str(c.get("desc", ""))))


def _add_line(script: ClipScript, ln: dict, duration: float | None, declared: set[str]) -> None:
    raw_who = str(ln.get("who", "")).strip()
    kind = str(ln.get("kind", "")).strip().lower()
    if kind in {"sound", "audio", "sfx", "noise"}:
        kind = "environment"
    # ENV is the reserved who for environment lines: either who=ENV or
    # kind=environment sends the line to environment, on the same principle as the
    # reserved SW id.
    if raw_who.upper() == ENV_WHO or kind == "environment":
        kind, who = "environment", ENV_WHO
    elif kind not in LINE_KINDS:
        raise ValueError(f"bad kind {kind!r}")
    else:
        who = raw_who
        if not _LOCAL_ID_RE.fullmatch(who):
            raise ValueError(f"bad who {who!r}")
        if who not in declared:
            raise ValueError(f"undeclared cast {who!r}")
    t0 = _clamp(_fnum(ln.get("t0", 0)), duration)
    t1 = _clamp(_fnum(ln.get("t1", 0)), duration)
    if t1 < t0:
        t0, t1 = t1, t0
    text = str(ln.get("text", "")).strip()
    if not text:
        raise ValueError("empty text")
    script.lines.append(ClipLine(t0=t0, t1=t1, who=who, kind=kind, text=text))


def _add_nom(script: ClipScript, n: dict, duration: float | None, declared: set[str]) -> None:
    local_id = str(n.get("id", "")).strip()
    if local_id not in declared:
        raise ValueError(f"undeclared cast {local_id!r}")
    pos = str(n.get("pos", "")).strip().lower()
    if pos not in _POS_VALUES:
        pos = ""   # missing or illegal position: keep the nomination, but disambiguate as if it had no position rather than guessing
    script.nominations.append(Nomination(local_id=local_id, t=_clamp(_fnum(n.get("t", 0)), duration),
                                         desc=str(n.get("desc", "")), pos=pos))


def _add_voice(script: ClipScript, v: dict, duration: float | None, declared: set[str]) -> None:
    local_id = str(v.get("id", "")).strip()
    if local_id not in declared:
        raise ValueError(f"undeclared cast {local_id!r}")
    t0 = _clamp(_fnum(v.get("t0", 0)), duration)
    t1 = _clamp(_fnum(v.get("t1", 0)), duration)
    if t1 <= t0:
        raise ValueError("empty range")
    script.voice_ranges.append(VoiceRange(local_id=local_id, t0=t0, t1=t1))


def _add_cont(script: ClipScript, ct: dict, declared: set[str]) -> None:
    local_id = str(ct.get("id", "")).strip()
    if local_id not in declared:
        raise ValueError(f"undeclared cast {local_id!r}")
    prev = str(ct.get("prev", "none")).strip()
    if not _PREV_RE.fullmatch(prev):
        prev = "none"
    script.cont[local_id] = prev
    if ct.get("evidence"):
        script.cont_evidence[local_id] = str(ct["evidence"])
