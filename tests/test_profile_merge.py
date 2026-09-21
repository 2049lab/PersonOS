"""Merge engine unit tests (pure logic, clean structures): applying patches, letting the LLM choose the band,
moving facts between bands, evicting as a fallback when a band overflows, and the over_cap report.
"""

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
    assert p.traits["personality"].text == "calm"                 # fields the patch did not mention are kept
    assert p.traits["occupation"].text == "engineer"
    assert p.traits["occupation"].last_confirmed == "2026-09-14"   # the engine stamps today's date


def test_trait_null_clears():
    cur = UserProfile.empty()
    cur.traits["finance"] = ProfileTrait(text="frugal", status="inferred",
                                        last_confirmed="2026-01-01", sources=["c0"])
    p = apply_patch(cur, {"traits": {"finance": None}}, today_str=TODAY)
    assert p.traits["finance"] is None


def test_fact_add_goes_to_declared_band():
    """The LLM states the band explicitly and the engine files the fact there as given, rather than assigning bands
    mechanically by date.
    """
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
    assert f.sources == ["c0", "cell_1"]              # sources are unioned so provenance survives
    assert f.last_confirmed == "2026-09-14"           # the fact was touched, so it is stamped with today


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
    """The week band holds at most 3: push in 4 facts with different last_confirmed dates and the oldest one is
    evicted, leaving the 3 most recent.
    """
    cur = UserProfile.empty()
    for i, lc in enumerate(["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13"]):
        cur.facts["week"].append(ProfileFact(id=f"f{i}", text=f"d{lc}", last_confirmed=lc, sources=["c0"]))
    enforce_caps(cur)
    texts = {f.text for f in cur.facts["week"]}
    assert len(texts) == 3 and "d2026-09-10" not in texts   # the oldest one was evicted


def test_over_cap_reports_overflowing_bands():
    cur = UserProfile.empty()
    cur.facts["today"].append(ProfileFact(id="a", text="1", last_confirmed=TODAY, sources=["c0"]))
    cur.facts["today"].append(ProfileFact(id="b", text="2", last_confirmed=TODAY, sources=["c0"]))
    assert over_cap(cur) == {"today": 2}              # today holds at most 1 and now has 2, so it is reported


def test_apply_patch_evict_false_leaves_overflow_for_bounce():
    """The consolidate loop passes evict=False: overflow is not evicted here but left for over_cap to detect, so the
    work can be bounced back to the LLM to converge.
    """
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
    assert cur.traits["personality"].text == "orig"    # the input argument is left untouched
