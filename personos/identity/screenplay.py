"""一路剧本 protocol:JSON schema prompt + 解析校验(合并对话/动作/环境为单路)。

为什么用 JSON 而非竖线定长文本:定长文本靠字段位置,模型少一段(如漏 kind)或用裸数字 id
就整行错位,得靠 parser 打补丁;JSON 键名显式,qwen 遵循度高、json.loads 解析稳,几乎不用兜底
(定案 §4 + 评审:尽量让 MLLM 直接输出对,不做补丁兜底)。

相对 mneme 的改造(定案 §4):
- **一路**产出:一个 Omni prompt 同时出 对话/动作/环境(line 的三种 kind),不分三路;
- **salience 门控**:只为有承载力的人建 cast(说话/与佩戴者互动/被喊名/显著动作),路人进环境行;
- **横向三分位 pos**(对齐 mneme):nom 的 pos 用 left/center/right(人在画面主要左右分开,不引纵向);
  素材归属挑框后再做一致性校验(挑中的脸实际三分位≠提名则拒,宁缺毋滥),见 harvest.pick_face。

输出:单个 JSON 对象 {casts, lines, noms, voices, conts}。字段见 build_clip_prompt 里的 schema。
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
    """把行文本里**裸露的人物 id** 换成 mapping 给的名字。

    为什么必须做:剧本 MLLM 写 action/environment 行时会在文本里直呼 id——
    "P1 enters holding a basketball"、"where P2 is seated"。归属改写只动 holder,
    这些 id 就一路漏进 evidence → episode → atom,同一个人在记忆里出现三种叫法
    (holder 叫「人物#1」、文本里叫 P1、别人嘴里叫 Alice),检索和作答都分不清是同一人。

    分两段做,各在信息齐的地方:clip 处理时 local id(P1)→会话 cast id(S1,有 cast_map);
    会话末 flush 时 cast id(S1)→展示名(有终审归属)。

    只替换 mapping 里确实有的 key,且要求词边界——不碰正常词汇;长 key 优先,防 P1 把 P12 切一半。
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
_POS_VALUES = ("left", "center", "right")   # 横向三分位(对齐 mneme _pick_face)


# ── 数据结构(移植 mneme types 的剧本部分)──────────────────────────

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
    pos: str = ""                 # 横向三分位 left/center/right(多脸消歧;缺失=不按位置挑框)


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
    parsed_ok: bool = False       # JSON 成功解析(替代竖线协议的 saw_end)
    raw: str = ""
    cast_map: dict[str, str] = field(default_factory=dict)  # local_id -> 会话 cast,映射端回填

    def cast_decl(self, local_id: str) -> CastDecl | None:
        return next((c for c in self.casts if c.local_id == local_id), None)

    def session_cast_ids(self) -> list[str]:
        """本 clip 出现的会话 cast id(cast_map 值去重保序;映射端回填后可用)。"""
        seen: list[str] = []
        for cast_id in self.cast_map.values():
            if cast_id not in seen:
                seen.append(cast_id)
        return seen


# ── prompt(一路剧本 JSON schema;我们相对 mneme 的改造集中在此)──────

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
    """构造一路剧本 JSON prompt。返回 (prompt, roster 图 b64 列表)。

    scene_setting:调用方场景设定(眼镜第一人称 / 机器人 / 纪录片…),对齐场景理解。
    roster_cards:本会话早前出现的 cast(名字/外观/参考图),供续接判断,不是结论。
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


# ── 解析(JSON;只做校验,不修补格式)────────────────────────────────

def _extract_json(raw: str) -> str:
    """剥 markdown 围栏 + 取第一个 {...} 到最后一个 }(容忍模型偶尔加围栏/前后缀)。"""
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
    """解析 Omni 的 JSON 剧本。坏记录进 issues 跳过,不级联;整体 JSON 坏则 parsed_ok=False。"""
    script = ClipScript(raw=raw)
    try:
        data = json.loads(_extract_json(raw))
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except Exception as exc:  # noqa: BLE001
        script.issues.append(f"JSON 解析失败: {type(exc).__name__}: {exc}")
        return script
    script.parsed_ok = True

    for i, c in enumerate(data.get("casts") or []):
        try:
            _add_cast(script, c)
        except Exception as exc:  # noqa: BLE001
            script.issues.append(f"cast[{i}]: {exc}")
    declared = {c.local_id for c in script.casts} | {WEARER_CAST_ID}  # SW 是保留的佩戴者 id,恒合法
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
    # SW(佩戴者)被引用却没显式建 cast → 补一条(SW 是结构性保留 id,非格式补丁)
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
    # ENV 是保留 who(环境行):who=ENV 或 kind=environment 一律归环境(与 SW 保留 id 同理)
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
        pos = ""   # 缺失/非法位置:保留提名,多脸消歧按"无位置"(宁缺毋滥)
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
