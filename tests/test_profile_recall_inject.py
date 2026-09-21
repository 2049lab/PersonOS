"""Unit tests for the profile consumer side: the public structured /profile view (a pure
function) plus R0/R5 injection (asserted by capturing the messages)."""

from __future__ import annotations

from datetime import datetime

from personos.models import MemCell, now
from personos.online.retrieval import CellHit, answer_from_cells, rewrite_query
from personos.storage.profile_store import (
    ProfileFact,
    ProfileTrait,
    ProfileVersion,
    UserProfile,
)


class CapturingLLM:
    """Records the last messages it was given and returns a canned response, so injection can
    be verified."""

    def __init__(self, resp: str):
        self.resp = resp
        self.messages = None

    def chat(self, messages, temperature=0.3, max_tokens=2048) -> str:
        self.messages = messages
        return self.resp


def _seeded() -> ProfileVersion:
    p = UserProfile.empty()
    p.traits["communication_style"] = ProfileTrait(
        text="direct", status="confirmed", last_confirmed="2026-09-01", sources=["cell_x"])
    p.facts["today"].append(ProfileFact(
        id="f1", text="2026-09-14 had ramen", last_confirmed="2026-09-14", sources=["cell_y"]))
    return ProfileVersion(version=3, profile=p, up_to_cell_id="cell_y",
                          created_at=datetime(2026, 9, 14, 8, 30))


def test_public_profile_empty():
    from server.api import _public_profile
    v = _public_profile(None)
    assert v["exists"] is False and v["version"] == 0 and v["traits"] == {}


def test_public_profile_seeded_strips_internal_ids():
    from server.api import _public_profile
    v = _public_profile(_seeded())
    assert v["exists"] is True and v["version"] == 3
    t = v["traits"]["communication_style"]
    assert t == {"text": "direct", "status": "confirmed", "last_confirmed": "2026-09-01"}  # no sources
    f = v["facts"]["today"][0]
    assert f == {"text": "2026-09-14 had ramen", "last_confirmed": "2026-09-14"}  # no id, no sources


def test_rewrite_query_injects_profile():
    llm = CapturingLLM('{"resolved": "how is Alice"}')
    rewrite_query(llm, raw_query="how is she", now_dt=now(),
                  profile="USER PROFILE:\n[基本特征]\n家庭: wife is Alice(确认)")
    user_msg = llm.messages[1]["content"]
    assert "USER PROFILE" in user_msg and "Alice" in user_msg   # R0 injected the profile


def test_rewrite_query_without_profile_has_no_block():
    llm = CapturingLLM('{"resolved": "q"}')
    rewrite_query(llm, raw_query="q", now_dt=now())            # empty profile
    # With no profile, behaviour is the same as it is today.
    assert "USER PROFILE" not in llm.messages[1]["content"]


def test_answer_from_cells_injects_profile():
    hit = CellHit(cell=MemCell(session_id="s", topic="t", episode="user did X", t_start=now()),
                  score=1.0, best_sim=1.0)
    llm = CapturingLLM('{"answer": "ok", "cells": ["m1"]}')
    answer_from_cells(llm, query="q", subject="user", hits=[hit], now_dt=now(),
                      profile="USER PROFILE:\n[基本特征]\n沟通风格: 偏好简短(确认)")
    user_msg = llm.messages[1]["content"]
    assert "USER PROFILE" in user_msg and "偏好简短" in user_msg   # R5 injected the traits
