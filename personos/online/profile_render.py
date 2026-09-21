"""Profile rendering: a structured UserProfile -> the plain-text block injected internally (R0 / R5 /
deep track).

Different from the public /profile endpoint, which returns structured JSON only; the text block here
is fed into the R0 / R5 / deep-track prompts and is rendered deterministically from the structure
(the date prefix and the confirmed/inferred markers are products of rendering, never written by hand
by an agent). It carries a disclaimer: the profile may guide "where to look and how to answer", but
the factual basis must still come from the memory materials (the boundary it shares with R3'
adjudication).
"""

from __future__ import annotations

from datetime import datetime

from ..storage.profile_store import BANDS, PMO16, UserProfile

_DISCLAIMER = ("USER PROFILE(关于提问用户的已知信息,仅用于理解问题与组织回答,"
               "不可当作回答事实依据):")

# English key -> Chinese label (the injected text goes to the answering / rewriting LLM, and reads
# better in a Chinese context)
_KEY_LABELS = {
    "personality": "性格", "communication_style": "沟通风格", "social_style": "社交倾向",
    "occupation": "职业", "goals": "目标", "values": "价值观", "work_style": "工作模式",
    "learning_style": "学习模式", "tech_environment": "技术环境", "lifestyle": "生活方式",
    "health": "健康", "finance": "财务",
    "identity": "身份", "location": "地域", "family": "家庭", "interests": "兴趣",
}
_STATUS_CN = {"confirmed": "确认", "inferred": "推断"}


def _fmt_date(s: str) -> str:
    """2026-09-10 -> the Chinese date form; returned unchanged when it cannot be parsed."""
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return f"{d.year}年{d.month:02d}月{d.day:02d}日"
    except (ValueError, TypeError):
        return s or "?"


def render(profile: UserProfile | None, *, mode: str = "full") -> str:
    """Render the injection block. mode='full' (traits + facts, for R0 and the deep track) |
    'traits' (traits only, for R5). An empty profile renders as ''."""
    if profile is None:
        return ""
    lines: list[str] = []

    trait_lines = []
    for key in PMO16:                                    # fixed order (L1 -> L3), stable and unit-testable
        t = profile.traits.get(key)
        if t and t.text:
            lc = f",印证于{_fmt_date(t.last_confirmed)}" if t.last_confirmed else ""
            trait_lines.append(f"{_KEY_LABELS.get(key, key)}: {t.text}({_STATUS_CN.get(t.status, t.status)}{lc})")
    if trait_lines:
        lines.append("[基本特征]")
        lines.extend(trait_lines)

    if mode == "full":
        fact_lines = []
        for band in BANDS:                               # today -> long
            for f in profile.facts.get(band, []):
                if f.text:
                    fact_lines.append(f"- {f.text}")     # the date is already in the body text, so rendering adds no prefix
        if fact_lines:
            lines.append("[近期事实]")
            lines.extend(fact_lines)

    if not lines:                                        # nothing at all -> empty string (the consumer's None path)
        return ""
    return _DISCLAIMER + "\n" + "\n".join(lines)
