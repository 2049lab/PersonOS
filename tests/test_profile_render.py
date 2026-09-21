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
    assert "USER PROFILE" in out and "never cite it as evidence" in out   # the disclaimer line
    assert "[Traits]" in out and "communication style: direct and terse(inferred, last confirmed 2026-09-01)" in out
    assert "[Recent facts]" in out and "- 2026-09-14 had ramen" in out   # the date lives in the body text


def test_traits_mode_omits_facts():
    out = render(_profile(), mode="traits")
    assert "[Traits]" in out
    assert "[Recent facts]" not in out and "ramen" not in out


def test_confirmed_label():
    p = UserProfile.empty()
    p.traits["identity"] = ProfileTrait(text="engineer", status="confirmed",
                                        last_confirmed="2026-09-10", sources=["c1"])
    out = render(p, mode="traits")
    assert "identity: engineer(confirmed, last confirmed 2026-09-10)" in out
