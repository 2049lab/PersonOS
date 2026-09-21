"""剧本物理矛盾的**统一重修**:一次检出全部 → 一次性交回模型改 → 仍不过则保守降级。

为什么要有这一层:MLLM 会幻觉,而"一个人被当成两个角色""同一个人同时在两个位置"这类
**物理上不可能**的错误,用代码判比让模型自查可靠得多。检出后不是直接丢数据,而是把
**所有**矛盾连同它自己上次的输出一起喂回去让它改 —— 明显错了就该重评,这是 mneme 的做法。

与 mneme 的一处有意简化:mneme 分两条重修回路(cast 类在 harvest 前、voice 类在 harvest 后,
因为它的 voice 重修 prompt 要用抽好的脸照)。我们**合并成一次、全放在 harvest 之前**:
- 规则 1/2/3/5/7 全是逻辑矛盾,不是"看脸判断",有视频 + roster 卡就够;
- 一次调用而不是两次(每次都要重发视频,在我们的部署上 ~100s,省一次是一次);
- 坏提名在**抽脸之前**就被修好,错脸根本没机会进概率云。

流程(≤2 轮,对齐 mneme MAX_REPAIR_ATTEMPTS):
    检测 → 构造[上次输出 + 矛盾清单] → 重跑 → 合并(引用完整性校验) → 重新检测
    → 干净则采纳;仍有矛盾则**把部分改善也采纳为新基线**带着剩余矛盾再试一轮
    → 两轮后仍不过 → 保守降级(只做减法)

降级的取向恒为「宁缺毋滥」:丢掉可疑的那条提名/语音段/名字,绝不凭空添加归属。
误伤的代价有界且可恢复(后续 clip 会再提名同一个人);而错误归属一旦学进概率云不可逆。
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
_MIN_VOICE_SEC = 0.4       # 与 harvest._MIN_VOICE_SEC / inspect 对齐

# 三档:degrade=检测+重修+降级(默认) / detect=只检测只留痕不改剧本 / off=全关
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
    """把模型上次的 casts/noms/voices/conts 原样回灌 —— 让它在自己的输出上改,而不是重写。"""
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
    """[规则头 + roster + 上次输出 + 保留的台词 + 矛盾清单] —— 一次把所有问题交出去。"""
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
    """把重修输出合并回原剧本(只换 casts/noms/voices/conts,lines 原样保留)。

    **引用完整性校验**:保留的台词与新的 voices 引用的每个 id 都必须仍在 casts 里,
    否则整次修复作废返回 None —— 台词归属不能悬空,宁可退回上一版。
    """
    # 复用主解析器:重修输出与剧本同 schema(缺 lines),解析后把 lines 换回原来的
    patched = parse_clip_output(raw, duration_sec=duration_sec)
    if not patched.parsed_ok or not patched.casts:
        return None
    ids = {c.local_id for c in patched.casts} | {WEARER_CAST_ID, ENV_WHO}
    for line in script.lines:
        if line.who not in ids:
            logger.warning(f"重修作废:保留的台词引用了不存在的 id {line.who!r}")
            return None
    for vr in patched.voice_ranges:
        if vr.local_id not in ids:
            logger.warning(f"重修作废:voice 引用了不存在的 id {vr.local_id!r}")
            return None
    return ClipScript(
        casts=patched.casts, lines=list(script.lines), nominations=patched.nominations,
        voice_ranges=patched.voice_ranges, cont=patched.cont,
        cont_evidence=patched.cont_evidence,
        issues=[*script.issues, *patched.issues], parsed_ok=True, raw=raw)


def degrade(script: ClipScript, violations: list[Violation]) -> ClipScript:
    """保守降级:**只做减法**。丢掉可疑记录,绝不凭空添加归属。

    误伤代价有界且可恢复(后续 clip 会再提名同一个人、名字还在 roster 里);
    而错误归属一旦学进概率云就不可逆 —— 所以取向恒为宁缺毋滥。
    """
    casts, noms = list(script.casts), list(script.nominations)
    voices, cont = list(script.voice_ranges), dict(script.cont or {})
    hit = {v.rule for v in violations}
    notes: list[str] = []

    if "duplicate_cast" in hit:                       # 同 id 多次声明 → 只保留第一条
        seen, kept = set(), []
        for c in casts:
            if c.local_id in seen:
                notes.append(f"弃重复声明 {c.local_id}")
                continue
            seen.add(c.local_id); kept.append(c)
        casts = kept

    if "cont_conflict" in hit:                        # 断开隐含续接,交给仲裁凭素材判
        for v in violations:
            if v.rule != "cont_conflict":
                continue
            for cid in v.cast_ids:
                if cont.pop(cid, None) is not None:
                    notes.append(f"断开续接 {cid}")

    drop_t: dict[str, set[float]] = {}                # 分身提名 / 佩戴者提名 → 弃掉
    for v in violations:
        if v.rule in ("nom_position_conflict", "wearer_visible"):
            for cid in v.cast_ids:
                drop_t.setdefault(cid, set()).update(v.times)
    if drop_t:
        before = len(noms)
        noms = [n for n in noms
                if not any(abs(n.t - t) < 1e-6 for t in drop_t.get(n.local_id, ()))]
        if len(noms) != before:
            notes.append(f"弃提名 {before - len(noms)} 条")

    for v in violations:                              # 语音重叠 → 双方都不采重叠段
        if v.rule != "voice_overlap" or len(v.times) != 2:
            continue
        lo, hi = v.times
        cut: list[VoiceRange] = []
        for r in voices:
            if r.local_id not in v.cast_ids or r.t1 <= lo or r.t0 >= hi:
                cut.append(r)
                continue
            for a, b in ((r.t0, lo), (hi, r.t1)):     # 只留不重叠且够长的残段
                if b - a >= _MIN_VOICE_SEC:
                    cut.append(replace(r, t0=a, t1=b))
            notes.append(f"裁语音 {r.local_id}[{r.t0:.1f},{r.t1:.1f}]")
        voices = cut

    for v in violations:                              # 名字无台词支撑 → 剥名(roster 里还有)
        if v.rule not in DEGRADE_ONLY:
            continue
        for cid in v.cast_ids:
            casts = [replace(c, name=None, name_evidence="none") if c.local_id == cid else c
                     for c in casts]
            notes.append(f"剥名 {cid}")

    return ClipScript(
        casts=casts, lines=list(script.lines), nominations=noms, voice_ranges=voices,
        cont=cont, cont_evidence=dict(script.cont_evidence),
        issues=[*script.issues, *(f"degraded: {n}" for n in notes)],
        parsed_ok=script.parsed_ok, raw=script.raw)


def enforce(script: ClipScript, *, omni: Any = None, clip_url: str = "",
            roster_cards: str = "", duration_sec: Optional[float] = None,
            session_id: str = "", clip_index: int = 0) -> tuple[ClipScript, dict]:
    """统一检测 → 一次性重修(≤2 轮)→ 保守降级。返回 (剧本, 报告)。

    报告字段供上层打结构化日志与统计:检出/修好/残留各多少、按规则计数、重修跑了几轮。
    任何异常都不抛 —— 守护层坏了不该把整条 clip 拖垮,最坏退化成"不守护"。
    """
    rep: dict[str, Any] = {"mode": MODE, "found": {}, "attempts": 0,
                           "remaining": {}, "degraded": False}
    if MODE == "off":
        return script, rep
    try:
        return _enforce(script, rep, omni=omni, clip_url=clip_url, roster_cards=roster_cards,
                        duration_sec=duration_sec)
    except Exception as e:  # noqa: BLE001  守护层自己坏了,最坏退化成"不守护",不拖垮整条 clip
        logger.exception(f"一致性守护异常,本 clip 不守护 session={session_id} clip={clip_index}: {e}")
        rep["error"] = f"{type(e).__name__}: {e}"
        return script, rep


def _enforce(script: ClipScript, rep: dict, *, omni: Any, clip_url: str,
             roster_cards: str, duration_sec: Optional[float]) -> tuple[ClipScript, dict]:
    violations = inspect_script(script)
    rep["found"] = _by_rule(violations)
    if not violations:
        return script, rep
    if MODE == "detect":       # 只留痕不改剧本:上线初期收集触发率用
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
            except Exception as e:  # noqa: BLE001  重修失败退化成降级,不拖垮 clip
                logger.warning(f"剧本重修第 {attempt} 轮失败,转降级: {e}")
                break
            if patched is None:
                continue
            left = inspect_script(patched)
            # 部分改善也采纳为新基线(引用完整性已校验过),带着剩余矛盾再试一轮
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
