"""画像渲染:结构化 UserProfile → 内部注入用的纯文本块(R0/R5/深轨)。

与对外 /profile 接口不同——对外只给结构化 JSON;这里的文本块是喂给 R0/R5/深轨 prompt 的,
从结构确定性渲染(日期前缀、确认/推断标记都是渲染产物,agent 不手写)。带免责声明:画像可
引导"去哪找、怎么答",事实依据仍须来自记忆材料(与 R3' 核判共存的边界)。
"""

from __future__ import annotations

from datetime import datetime

from ..storage.profile_store import BANDS, PMO16, UserProfile

_DISCLAIMER = ("USER PROFILE(关于提问用户的已知信息,仅用于理解问题与组织回答,"
               "不可当作回答事实依据):")

# 英文 key → 中文标签(注入文本给作答/改写 LLM 用,中文语境更好读)
_KEY_LABELS = {
    "personality": "性格", "communication_style": "沟通风格", "social_style": "社交倾向",
    "occupation": "职业", "goals": "目标", "values": "价值观", "work_style": "工作模式",
    "learning_style": "学习模式", "tech_environment": "技术环境", "lifestyle": "生活方式",
    "health": "健康", "finance": "财务",
    "identity": "身份", "location": "地域", "family": "家庭", "interests": "兴趣",
}
_STATUS_CN = {"confirmed": "确认", "inferred": "推断"}


def _fmt_date(s: str) -> str:
    """2026-09-10 → 2026年09月10日;不可解析原样返回。"""
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return f"{d.year}年{d.month:02d}月{d.day:02d}日"
    except (ValueError, TypeError):
        return s or "?"


def render(profile: UserProfile | None, *, mode: str = "full") -> str:
    """渲染注入块。mode='full'(traits+facts,R0/深轨)| 'traits'(仅特征,R5)。空画像 → ''。"""
    if profile is None:
        return ""
    lines: list[str] = []

    trait_lines = []
    for key in PMO16:                                    # 固定顺序(L1→L3),稳定可单测
        t = profile.traits.get(key)
        if t and t.text:
            lc = f",印证于{_fmt_date(t.last_confirmed)}" if t.last_confirmed else ""
            trait_lines.append(f"{_KEY_LABELS.get(key, key)}: {t.text}({_STATUS_CN.get(t.status, t.status)}{lc})")
    if trait_lines:
        lines.append("[基本特征]")
        lines.extend(trait_lines)

    if mode == "full":
        fact_lines = []
        for band in BANDS:                               # today→long
            for f in profile.facts.get(band, []):
                if f.text:
                    fact_lines.append(f"- {f.text}")     # 日期已写在正文,渲染不再拼前缀
        if fact_lines:
            lines.append("[近期事实]")
            lines.extend(fact_lines)

    if not lines:                                        # 无任何内容 → 空串(消费侧 None 路径)
        return ""
    return _DISCLAIMER + "\n" + "\n".join(lines)
