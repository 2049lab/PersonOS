"""物理一致性检查:检测 Omni 输出里**现实世界不可能**的自相矛盾。

MLLM 会幻觉,而幻觉里最该被机器挡住的是这类"物理上不可能"的归属错误——
一个人被当成两个角色、同框的两个人被当成同一个人、一个人同时出现在两个位置。
这些用代码判比让模型自查可靠得多,所以先机器检出,再定向让模型重改(见 repair.py)。

规则(编号对齐 mneme anchor/inspect.py,便于两边对照):
  1 voice_overlap        两人语音区间相交 —— 声纹会被混进一个模板
  2 cont_conflict        两个 cast 续接同一昔日成员 —— 一人不能变两人
  3 nom_position_conflict同一 cast 瞬时被提名在两个位置 —— 人不会分身
  5 duplicate_cast       同一 id 声明多次
  6 bind_collision       同框 cast 绑到同一 character(仲裁阶段判,需 proposed)
  7 wearer_visible       佩戴者被提名出现在画面里 —— 第一人称拍不到自己
  8 name_unsupported     名字在本 clip 任何台词里都没出现,却声称台词类证据
  9 name_self_address    名字只出现在他自己台词里 —— 人不会喊自己的名字

**只检测,纯函数无副作用**;重修/降级在 repair.py,仲裁期的 6 号重裁在 chains.py。
"""

from __future__ import annotations

from dataclasses import dataclass

from personos.identity.screenplay import ENV_WHO, WEARER_CAST_ID, ClipScript


@dataclass(frozen=True)
class Violation:
    """一处物理矛盾:机器规则名 + 涉及 cast + 人类可读细节。"""

    rule: str
    cast_ids: tuple[str, ...]
    detail: str
    times: tuple[float, ...] = ()


def present_casts(script: ClipScript) -> set[str]:
    """本 clip 真正在场的会话 cast(有台词/提名/语音),空 CAST 壳不算同框。"""
    present: set[str] = set()
    for line in script.lines:
        if line.who != ENV_WHO:
            cast_id = script.cast_map.get(line.who)
            if cast_id:
                present.add(cast_id)
    for nom in script.nominations:
        cast_id = script.cast_map.get(nom.local_id)
        if cast_id:
            present.add(cast_id)
    for vr in script.voice_ranges:
        cast_id = script.cast_map.get(vr.local_id)
        if cast_id:
            present.add(cast_id)
    return present


def inspect_bind_collisions(script: ClipScript, proposed: dict[str, str]) -> list[Violation]:
    """Rule 6:同框 cast 撞到同一 character。

    proposed: {会话 cast_id -> character_id}(提交前的试探绑定;NEW/空值排除——新档独立不碰撞)。
    佩戴者与空壳 cast 也排除。
    """
    present = present_casts(script)
    groups: dict[str, list[str]] = {}
    for cast_id, character_id in proposed.items():
        if cast_id == WEARER_CAST_ID or not character_id or character_id == "NEW":
            continue
        if cast_id not in present:
            continue
        groups.setdefault(character_id, []).append(cast_id)
    return [
        Violation(
            rule="bind_collision", cast_ids=tuple(sorted(members)),
            detail=(f"co-occurring casts {', '.join(sorted(members))} all bound to "
                    f"{character_id} — people appearing together in one clip are"
                    " physically distinct"),
        )
        for character_id, members in groups.items()
        if len(members) > 1
    ]


# ── 规则 1/2/3/5/7:剧本内部的物理矛盾(解析后即可查,不依赖 cast_map)────────

_NOM_SAME_MOMENT_SEC = 0.5   # 提名时刻差在此以内视为"同一瞬间"(对齐 mneme)
_MIN_VOICE_SEC = 0.4         # 与 harvest._MIN_VOICE_SEC 对齐:短于此的残段无声纹价值


def inspect_voice_overlap(script: ClipScript) -> list[Violation]:
    """Rule 1:两个**不同** cast 的语音区间相交。

    prompt 明确要求"只给这个人单独说话的区间",所以相交 = 模型违反了自己的契约,
    至少一条是错的。不修的话两个人的声音会被混进同一个声纹模板,污染不可逆。
    """
    out: list[Violation] = []
    rs = [v for v in script.voice_ranges if v.local_id != ENV_WHO]
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            if a.local_id == b.local_id:
                continue
            lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
            if hi <= lo:
                continue
            out.append(Violation(
                rule="voice_overlap", cast_ids=tuple(sorted((a.local_id, b.local_id))),
                detail=(f"voice ranges overlap: {a.local_id} [{a.t0:.1f},{a.t1:.1f}] vs "
                        f"{b.local_id} [{b.t0:.1f},{b.t1:.1f}] — a voice range must cover "
                        f"only that person speaking ALONE"),
                times=(lo, hi)))
    return out


def inspect_cont_conflict(script: ClipScript) -> list[Violation]:
    """Rule 2:两个 cast 都声称续接同一个昔日成员 —— 一个人不能变成两个人。"""
    by_prev: dict[str, list[str]] = {}
    for local_id, prev in (script.cont or {}).items():
        if prev and prev != "none":
            by_prev.setdefault(prev, []).append(local_id)
    return [
        Violation(rule="cont_conflict", cast_ids=tuple(sorted(ids)),
                  detail=(f"{', '.join(sorted(ids))} all continue roster member {prev} — "
                          f"one earlier person can continue as at most ONE cast here"))
        for prev, ids in by_prev.items() if len(ids) > 1
    ]


def inspect_nom_position(script: ClipScript) -> list[Violation]:
    """Rule 3:同一 cast 在同一瞬间被提名在两个横向位置 —— 人不会分身。

    至少有一个提名指错了人,照它抽脸就会把**别人的脸**塞进这个人的档案,
    再学进概率云后此后所有认人都受影响 —— 所以这条比看起来严重。
    """
    out: list[Violation] = []
    by_cast: dict[str, list] = {}
    for nom in script.nominations:
        if nom.pos:
            by_cast.setdefault(nom.local_id, []).append(nom)
    for local_id, noms in by_cast.items():
        noms = sorted(noms, key=lambda n: n.t)
        for i, a in enumerate(noms):
            for b in noms[i + 1:]:
                if b.t - a.t > _NOM_SAME_MOMENT_SEC:
                    break
                if a.pos != b.pos:
                    out.append(Violation(
                        rule="nom_position_conflict", cast_ids=(local_id,),
                        detail=(f"{local_id} nominated at {a.pos} (t={a.t:.1f}) and "
                                f"{b.pos} (t={b.t:.1f}) within "
                                f"{_NOM_SAME_MOMENT_SEC}s — one person cannot be in two places"),
                        times=(a.t, b.t)))
    return out


def inspect_duplicate_cast(script: ClipScript) -> list[Violation]:
    """Rule 5:同一 id 被声明多次(解析端只取第一条,重复声明多半是模型自相矛盾)。"""
    seen: dict[str, int] = {}
    for c in script.casts:
        seen[c.local_id] = seen.get(c.local_id, 0) + 1
    return [Violation(rule="duplicate_cast", cast_ids=(cid,),
                      detail=f"cast {cid} declared {n} times — declare each person once")
            for cid, n in seen.items() if n > 1]


def inspect_wearer_visible(script: ClipScript) -> list[Violation]:
    """Rule 7:佩戴者被提名出现在画面里 —— 第一人称机位拍不到自己,那张脸一定是别人的。"""
    times = tuple(n.t for n in script.nominations if n.local_id == WEARER_CAST_ID)
    if not times:
        return []
    return [Violation(
        rule="wearer_visible", cast_ids=(WEARER_CAST_ID,),
        detail=(f"wearer {WEARER_CAST_ID} nominated as visible at "
                f"{', '.join(f'{t:.1f}s' for t in times)} — the camera wearer is never"
                f" in frame; that face belongs to someone else"),
        times=times)]


def inspect_name_claims(script: ClipScript) -> list[Violation]:
    """Rule 8/9:名字与转写对账 —— 纯 MLLM 幻觉防线。

    8 name_unsupported :声称名字来自台词,但本 clip 任何台词里都没出现这个名字
                        (实测失败模式:模型复读 roster 里的已知名并伪造 explicit_dialogue)。
    9 name_self_address:名字只出现在他自己的台词里,且不是自我介绍 —— 人不会喊自己的名字。
    visible_text 豁免(画面文字无法与转写对账)。
    """
    spoken_by: dict[str, str] = {}
    for line in script.lines:
        if line.kind == "speech" and line.text:
            spoken_by[line.who] = spoken_by.get(line.who, "") + " " + line.text
    all_speech = " ".join(spoken_by.values())
    out: list[Violation] = []
    for c in script.casts:
        name = (c.name or "").strip()
        ev = (c.name_evidence or "none").lower()
        if not name or ev in ("none", "visible_text"):
            continue
        if name.lower() not in all_speech.lower():
            out.append(Violation(
                rule="name_unsupported", cast_ids=(c.local_id,),
                detail=(f"{c.local_id} claims name {name!r} with evidence {ev!r}, but "
                        f"{name!r} never appears in any speech line of this clip")))
            continue
        if ev == "self_introduction":
            continue
        others = " ".join(t for who, t in spoken_by.items() if who != c.local_id)
        if name.lower() not in others.lower():
            out.append(Violation(
                rule="name_self_address", cast_ids=(c.local_id,),
                detail=(f"{name!r} appears only in {c.local_id}'s own lines and is not a"
                        f" self-introduction — people do not call themselves by name")))
    return out


# 可由 repair 重修的规则(逻辑矛盾,模型重看一遍有机会改对)
REPAIRABLE = ("voice_overlap", "cont_conflict", "nom_position_conflict",
              "duplicate_cast", "wearer_visible")
# 只降级不重修:名字有无台词支撑,代码比模型可靠;重评大概率复读原答案(对齐 mneme)
DEGRADE_ONLY = ("name_unsupported", "name_self_address")


def inspect_script(script: ClipScript) -> list[Violation]:
    """剧本级统一检测(规则 1/2/3/5/7/8/9)。规则 6 需绑定结果,在仲裁阶段单独查。"""
    return [*inspect_voice_overlap(script), *inspect_cont_conflict(script),
            *inspect_nom_position(script), *inspect_duplicate_cast(script),
            *inspect_wearer_visible(script), *inspect_name_claims(script)]
