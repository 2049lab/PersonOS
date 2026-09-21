"""画像合并引擎(确定性,无 LLM):把整理产的**补丁**合并进当前画像。

干净结构下的分工:**带由 LLM 显式决定**(add/rewrite 里给 band),引擎不按日期机械分带;
引擎只做:应用补丁、盖 last_confirmed、**兜底各带条数上限**。

- traits:按域整替,未提域保留,None 显式清空;last_confirmed 引擎盖今天。
- facts:drop 按 id 删 / rewrite 按 id 打补丁(未提字段保留,sources 求并保溯源)/ add 引擎分配 f_id;
  被 touch 的 fact last_confirmed 盖今天;放哪个带听 LLM 的(band 字段)。
- 超限:各带条数上限,超了踢 last_confirmed 最旧的(consolidate 会先给 LLM 机会语义收敛,
  这里是最终兜底)。

补丁在合并前已由 profile_harness 校验字段合法、且 sources 已短→长回填(见 profile_consolidate)。
"""

from __future__ import annotations

from ..storage.profile_store import BANDS, ProfileFact, ProfileTrait, UserProfile

# 各带条数上限(见设计):超限踢 last_confirmed 最旧的
BAND_CAPS: dict[str, int] = {"today": 1, "week": 3, "month": 5, "long": 21}


def apply_patch(current: UserProfile | None, patch: dict, *, today_str: str,
                evict: bool = True) -> UserProfile:
    """把补丁合并进当前画像,返回新画像(不改入参)。

    today_str: YYYY-MM-DD,引擎给被 touch 的条目盖 last_confirmed。
    evict: True 则顺带兜底淘汰超限(默认);consolidate 循环里用 False,先让 LLM 语义收敛。
    """
    p = current.model_copy(deep=True) if current is not None else UserProfile.empty()
    for b in BANDS:
        p.facts.setdefault(b, [])

    _apply_traits(p, patch.get("traits") or {}, today_str)
    _apply_facts(p, patch.get("facts") or {}, today_str)
    if evict:
        enforce_caps(p)
    return p


def _apply_traits(p: UserProfile, traits_patch: dict, today_str: str) -> None:
    for dom, val in traits_patch.items():
        if val is None:
            p.traits[dom] = None                    # 显式清空(合法)
        else:
            p.traits[dom] = ProfileTrait(
                text=val.get("text", ""),
                status=val.get("status", "inferred"),
                last_confirmed=today_str,           # 引擎盖戳,不信 LLM 手写
                sources=list(val.get("sources") or []),
            )


def _find(p: UserProfile, fid: str) -> tuple[str, ProfileFact] | None:
    for b in BANDS:
        for f in p.facts[b]:
            if f.id == fid:
                return b, f
    return None


def _apply_facts(p: UserProfile, facts_patch: dict, today_str: str) -> None:
    for fid in facts_patch.get("drop") or []:
        hit = _find(p, fid)
        if hit:
            b, f = hit
            p.facts[b].remove(f)

    for rw in facts_patch.get("rewrite") or []:
        hit = _find(p, rw.get("id"))
        if not hit:
            continue                                # 未知 id:静默跳过(harness 已校验,兜底不炸)
        b, f = hit
        if "text" in rw:
            f.text = rw["text"]
        if "sources" in rw:
            f.sources = sorted(set(f.sources) | set(rw["sources"] or []))   # 求并保溯源
        f.last_confirmed = today_str                # 被 touch → 盖今天
        new_band = rw.get("band")
        if new_band and new_band != b:              # LLM 指定移带(如升 long)
            p.facts[b].remove(f)
            p.facts[new_band].append(f)

    for a in facts_patch.get("add") or []:
        f = ProfileFact(text=a.get("text", ""), last_confirmed=today_str,
                        sources=list(a.get("sources") or []))
        p.facts[a.get("band", "today")].append(f)   # 带听 LLM 的(harness 已保证 band 合法)


def over_cap(profile: UserProfile) -> dict[str, int]:
    """返回超上限的带 → 当前条数(空=都不超)。供 consolidate 判是否打回 LLM 收敛。"""
    return {b: len(profile.facts.get(b, [])) for b in BANDS
            if len(profile.facts.get(b, [])) > BAND_CAPS[b]}


def enforce_caps(profile: UserProfile) -> None:
    """兜底:各带超限则踢 last_confirmed 最旧的,留最新 cap 条(保留原相对顺序)。"""
    for b, cap in BAND_CAPS.items():
        lst = profile.facts.get(b, [])
        if len(lst) <= cap:
            continue
        # 按 last_confirmed 降序(稳定:同日期保留先插入的)取前 cap 个 id,再按原序过滤
        keep = {f.id for f in sorted(lst, key=lambda f: f.last_confirmed, reverse=True)[:cap]}
        profile.facts[b] = [f for f in lst if f.id in keep]
