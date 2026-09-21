"""Renderer unit tests: an empty profile renders to an empty string, full mode includes both traits
and facts, traits mode omits the facts, and a fact's date appears in the body text."""

from __future__ import annotations

from personos.online.profile_render import render
from personos.storage.profile_store import ProfileFact, ProfileTrait, UserProfile


def _profile() -> UserProfile:
    p = UserProfile.empty()
    p.traits["communication_style"] = ProfileTrait(
        text="direct and terse", status="inferred", last_confirmed="2026-09-01", sources=["c1"])
    p.facts["today"].append(ProfileFact(
        id="f1", text="2026-09-14 had ramen", last_confirmed="2026-09-14", sources=["c1"]))
    return p


def test_none_and_empty_render_to_blank():
    assert render(None) == ""
    assert render(UserProfile.empty()) == ""         # no content means an empty string, so the caller takes the no-profile path


def test_full_render_has_traits_and_facts():
    out = render(_profile(), mode="full")
    assert "USER PROFILE" in out and "不可当作回答事实依据" in out   # the disclaimer line
    assert "[基本特征]" in out and "沟通风格: direct and terse(推断,印证于2026年09月01日)" in out
    assert "[近期事实]" in out and "- 2026-09-14 had ramen" in out   # the date lives in the body text


def test_traits_mode_omits_facts():
    out = render(_profile(), mode="traits")
    assert "[基本特征]" in out
    assert "[近期事实]" not in out and "ramen" not in out


def test_confirmed_label():
    p = UserProfile.empty()
    p.traits["identity"] = ProfileTrait(text="engineer", status="confirmed",
                                        last_confirmed="2026-09-10", sources=["c1"])
    out = render(p, mode="traits")
    assert "身份: engineer(确认,印证于2026年09月10日)" in out
