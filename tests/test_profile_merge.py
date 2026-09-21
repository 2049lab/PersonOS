"""合并引擎单测(纯逻辑,干净结构):补丁合并 / 带由 LLM 定 / 移带 / 超限兜底淘汰 / over_cap 报告。"""

from __future__ import annotations

from personos.online.profile_merge import apply_patch, enforce_caps, over_cap
from personos.storage.profile_store import ProfileFact, ProfileTrait, UserProfile

TODAY = "2026-09-14"


def test_trait_replace_preserves_others_and_stamps_date():
    cur = UserProfile.empty()
    cur.traits["personality"] = ProfileTrait(text="calm", status="inferred",
                                             last_confirmed="2026-01-01", sources=["c0"])
    patch = {"traits": {"occupation": {"text": "engineer", "status": "confirmed", "sources": ["cell_1"]}}}
    p = apply_patch(cur, patch, today_str=TODAY)
    assert p.traits["personality"].text == "calm"                 # 未提域保留
    assert p.traits["occupation"].text == "engineer"
    assert p.traits["occupation"].last_confirmed == "2026-09-14"   # 引擎盖今天


def test_trait_null_clears():
    cur = UserProfile.empty()
    cur.traits["finance"] = ProfileTrait(text="frugal", status="inferred",
                                        last_confirmed="2026-01-01", sources=["c0"])
    p = apply_patch(cur, {"traits": {"finance": None}}, today_str=TODAY)
    assert p.traits["finance"] is None


def test_fact_add_goes_to_declared_band():
    """带由 LLM 显式指定,引擎照放(不按日期机械分带)。"""
    patch = {"facts": {"add": [
        {"band": "today", "text": "2026-09-14 had ramen", "sources": ["cell_1"]},
        {"band": "week", "text": "worked Mon–Fri on project X", "sources": ["cell_1"]},
        {"band": "long", "text": "married in 2020", "sources": ["cell_1"]},
    ]}}
    p = apply_patch(None, patch, today_str=TODAY)
    assert [f.text for f in p.facts["today"]] == ["2026-09-14 had ramen"]
    assert [f.text for f in p.facts["week"]] == ["worked Mon–Fri on project X"]
    assert [f.text for f in p.facts["long"]] == ["married in 2020"]


def test_rewrite_patches_preserves_and_unions_sources():
    cur = UserProfile.empty()
    cur.facts["week"].append(ProfileFact(id="f1", text="old text",
                                         last_confirmed="2026-09-10", sources=["c0"]))
    patch = {"facts": {"rewrite": [{"id": "f1", "text": "merged narrative", "sources": ["cell_1"]}]}}
    p = apply_patch(cur, patch, today_str=TODAY)
    f = p.facts["week"][0]
    assert f.text == "merged narrative"
    assert f.sources == ["c0", "cell_1"]              # 求并保溯源
    assert f.last_confirmed == "2026-09-14"           # 被 touch → 盖今天


def test_rewrite_moves_band_promote_to_long():
    cur = UserProfile.empty()
    cur.facts["month"].append(ProfileFact(id="f1", text="key fact",
                                          last_confirmed="2026-08-20", sources=["c0"]))
    p = apply_patch(cur, {"facts": {"rewrite": [{"id": "f1", "band": "long"}]}}, today_str=TODAY)
    assert p.facts["month"] == []
    assert [f.text for f in p.facts["long"]] == ["key fact"]


def test_drop_removes_fact():
    cur = UserProfile.empty()
    cur.facts["today"].append(ProfileFact(id="f1", text="x", last_confirmed=TODAY, sources=["c0"]))
    p = apply_patch(cur, {"facts": {"drop": ["f1"]}}, today_str=TODAY)
    assert p.facts["today"] == []


def test_enforce_caps_evicts_oldest_last_confirmed():
    """week 上限 3:塞 4 条不同 last_confirmed → 踢最旧的那条,留最新 3。"""
    cur = UserProfile.empty()
    for i, lc in enumerate(["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13"]):
        cur.facts["week"].append(ProfileFact(id=f"f{i}", text=f"d{lc}", last_confirmed=lc, sources=["c0"]))
    enforce_caps(cur)
    texts = {f.text for f in cur.facts["week"]}
    assert len(texts) == 3 and "d2026-09-10" not in texts   # 最旧被淘汰


def test_over_cap_reports_overflowing_bands():
    cur = UserProfile.empty()
    cur.facts["today"].append(ProfileFact(id="a", text="1", last_confirmed=TODAY, sources=["c0"]))
    cur.facts["today"].append(ProfileFact(id="b", text="2", last_confirmed=TODAY, sources=["c0"]))
    assert over_cap(cur) == {"today": 2}              # today 上限 1,现 2 条 → 报超


def test_apply_patch_evict_false_leaves_overflow_for_bounce():
    """consolidate 循环用 evict=False:超限不淘汰,留给 over_cap 判、打回 LLM 收敛。"""
    patch = {"facts": {"add": [
        {"band": "today", "text": "a", "sources": ["cell_1"]},
        {"band": "today", "text": "b", "sources": ["cell_1"]},
    ]}}
    p = apply_patch(None, patch, today_str=TODAY, evict=False)
    assert len(p.facts["today"]) == 2 and over_cap(p) == {"today": 2}


def test_apply_patch_does_not_mutate_input():
    cur = UserProfile.empty()
    cur.traits["personality"] = ProfileTrait(text="orig", status="inferred",
                                             last_confirmed="2026-01-01", sources=["c0"])
    apply_patch(cur, {"traits": {"personality": {"text": "new", "status": "confirmed", "sources": ["cell_1"]}}},
                today_str=TODAY)
    assert cur.traits["personality"].text == "orig"    # 入参不被改
